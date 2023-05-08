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
import agents


FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 1337, 'Random seed')
flags.DEFINE_integer("max_distance", 5, 'Maximum distance in the cube')
flags.DEFINE_integer("batch_size", 10, 'Batch size')
flags.DEFINE_integer("train_freq", 1, 'Training frequency')
flags.DEFINE_integer("eval_freq", 10_000, 'Evaluation frequency')
flags.DEFINE_integer("num_eval_episodes", 10, 'Number of evaluation episodes')
flags.DEFINE_integer("total_steps", 1_000_000, 'Total steps to train for.')
flags.DEFINE_integer("min_buffer_size", 1_000, 'Minimum buffer size before training.')
flags.DEFINE_float("lr", 0.1, 'Learning rate')
flags.DEFINE_float("gamma", 0.99, 'Discount factor')
flags.DEFINE_float("counts", 0.1, 'Coefficient for counts-based bonus')


def main(dummy_arg):

  env = Rubiks2x2x2(max_distance=FLAGS.max_distance, maximize_reward=True, distant_restarts=False, random_restarts=True)
  eval_env = Rubiks2x2x2(max_distance=FLAGS.max_distance, maximize_reward=True, distant_restarts=False, random_restarts=True)

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

  agent = agents.Agent(num_states=env.num_states,
                       num_actions=env.num_actions,
                       batch_size=FLAGS.batch_size,
                       lr=FLAGS.lr,
                       counts=FLAGS.counts)
  obs = env.reset()

  for t in range(FLAGS.total_steps):
    action = agent.act(obs.info['state'])
    next_obs = env.step(action)

    # if next_obs.reward > 0. :
    #   print(1.)
    transition = replay.Transition(
        s_tm1=obs.info['state'],
        a_tm1=action,
        r_t=next_obs.reward,
        discount_t=(1. - next_obs.is_last) * FLAGS.gamma,
        s_t=next_obs.info['state'],
    )
    buffer.add(transition)
    obs = env.reset() if next_obs.is_last else next_obs

    if t > FLAGS.min_buffer_size and t % FLAGS.train_freq == 0:
      agent.train_step(params, buffer)

    if t > 1 and t % FLAGS.eval_freq == 0:

      eval_reward = 0.
      for _ in range(FLAGS.num_eval_episodes):
        eval_obs = eval_env.reset()
        while not eval_obs.is_last:
          eval_action = agent.greedy_act(eval_obs.info['state'])
          eval_obs = eval_env.step(eval_action)
          eval_reward += eval_obs.reward
      print(f'Time step {t} Success rate {eval_reward/FLAGS.num_eval_episodes:.2f} States visited {agent.num_states_visited}')


if __name__ == "__main__":
  app.run(main)