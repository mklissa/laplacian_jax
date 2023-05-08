import collections
from absl import flags
from absl import app

import optax
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk

from rl_lap.rubiks222 import Rubiks2x2x2
import replay


FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 1337, 'Random seed')
flags.DEFINE_integer("max_distance", 5, 'Maximum distance in the cube')
flags.DEFINE_integer("batch_size", 10, 'Batch size')
flags.DEFINE_integer("train_freq", 1, 'Training frequency')
flags.DEFINE_integer("eval_freq", 10_000, 'Evaluation frequency')
flags.DEFINE_integer("num_eval_episodes", 10, 'Number of evaluation episodes')
flags.DEFINE_integer("total_steps", 500_000, 'Total steps to train for.')
flags.DEFINE_integer("min_buffer_size", 1_000, 'Minimum buffer size before training.')
flags.DEFINE_float("lr", 0.01, 'Learning rate')
flags.DEFINE_float("gamma", 0.1, 'Discount factor')


def train_step(params, buffer):
  batch = buffer.sample(FLAGS.batch_size)
  q_tm1 = params[batch.s_tm1, batch.a_tm1]
  q_t = params[batch.s_t]
  td_errors = batch.r_t + batch.discount_t * np.max(q_t, axis=1) - q_tm1
  params[batch.s_tm1, batch.a_tm1] += FLAGS.lr * td_errors
  return params

def online_train(params, batch):
  q_tm1 = params[batch.s_tm1, batch.a_tm1]
  q_t = params[batch.s_t]
  td_errors = batch.r_t + batch.discount_t * np.max(q_t) - q_tm1
  params[batch.s_tm1, batch.a_tm1] += FLAGS.lr * td_errors
  return params

def main(dummy_arg):

  env = Rubiks2x2x2(max_distance=FLAGS.max_distance, max_reward=True, random_restarts=True)
  eval_env = Rubiks2x2x2(max_distance=FLAGS.max_distance, max_reward=True, random_restarts=True)

  random_state = np.random.RandomState(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  params = np.zeros((env.num_states, env.num_actions))
  counts = np.zeros((env.num_states))

  buffer = replay.TransitionReplay(int(1e6), 
                                    replay.Transition(
                                    s_tm1=None,
                                    a_tm1=None,
                                    r_t=None,
                                    discount_t=None,
                                    s_t=None,),
                                    random_state)

  obs = env.reset()
  for t in range(FLAGS.total_steps):
    action = np.random.randint(env.num_actions)
    next_obs = env.step(action)

    transition = replay.Transition(
        s_tm1=obs.info['state'],
        a_tm1=action,
        r_t=next_obs.reward,
        discount_t=(1. - next_obs.is_last) * FLAGS.gamma,
        s_t=next_obs.info['state'],
    )
    buffer.add(transition)
    # print(next_obs.reward, obs.info['state'], next_obs.info['state'], env.distances[env.admissible_states[obs.info['state']]], env.distances[env.admissible_states[next_obs.info['state']]])
    obs = env.reset() if next_obs.is_last else next_obs

    # params = online_train(params, transition)
    if t > FLAGS.min_buffer_size and t % FLAGS.train_freq == 0:
      params = train_step(params, buffer)

    if t > 1 and t % FLAGS.eval_freq == 0:
      eval_reward = 0.

      for _ in range(FLAGS.num_eval_episodes):
        eval_obs = eval_env.reset()
        while not eval_obs.is_last:
          eval_action = np.argmax(params[eval_obs.info['state']])
          eval_obs = eval_env.step(eval_action)
          eval_reward += eval_obs.reward
      print(f'Time step {t} Success rate {eval_reward/FLAGS.num_eval_episodes:.2f}')


if __name__ == "__main__":
  app.run(main)