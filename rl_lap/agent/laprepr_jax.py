import os
import logging
import collections

import numpy as np
import torch
from torch import optim
import optax
import jax
import haiku as hk
from haiku import nets
import jax.numpy as jnp
import pickle


from . import episodic_replay_buffer
from ..envs import actors
from ..tools import py_tools
from ..tools import torch_tools
from ..tools import flag_tools
from ..tools import summary_tools
from ..tools import timer_tools


Data = collections.namedtuple("Data", "s1 s2 reward s_neg s_neg_2")


def neg_loss_fn(neg_rep_x, neg_rep_y):
    loss = 0
    n_dim = neg_rep_x.shape[0]
    coeff_vector = list(np.arange(n_dim, 0, -1)) + [0]
    # coeff_vector = list(1. / np.arange(1, n_dim + 1)) + [0]
    # coeff_vector = list(1. / 2 ** np.arange(n_dim)) + [0]

    for dim in range(n_dim, 0, -1):
        coeff = coeff_vector[dim-1] - coeff_vector[dim]
        loss += coeff * ( jnp.dot(neg_rep_x[:dim], neg_rep_y[:dim])**2 - jnp.dot(neg_rep_x[:dim], neg_rep_x[:dim]) / n_dim - jnp.dot(neg_rep_y[:dim], neg_rep_y[:dim]) / n_dim )
        # break
    return loss


def generalized_graph_drawing_loss_haiku(pos_rep_i, pos_rep_j, neg_rep, neg_rep_2, reward, beta=2.0):
    coeff_vector = jnp.arange(pos_rep_i.shape[1], 0, -1)
    # coeff_vector = 1. / jnp.arange(1, pos_rep_i.shape[1] + 1)
    # coeff_vector = 1. / 2**jnp.arange(pos_rep_i.shape[1])
    pos_loss = ((pos_rep_i - pos_rep_j)**2).dot(coeff_vector).mean()
    # pos_loss = 0

    neg_loss_vmap = jax.vmap(neg_loss_fn)
    neg_loss  = neg_loss_vmap(neg_rep, neg_rep_2).mean()

    loss = pos_loss + beta * neg_loss
    return loss, pos_loss, neg_loss



class LapReprLearner:

    @py_tools.store_args
    def __init__(self,
            d,
            # pytorch
            device=None,
            # env args
            action_spec=None,
            obs_shape=None,
            obs_prepro=None,
            env_factory=None,
            # learner args
            model_cfg=None,
            optimizer_cfg=None,
            n_samples=10000,
            batch_size=128,
            discount=0.0,
            w_neg=1.0,
            c_neg=1.0,
            reg_neg=0.0,
            replay_buffer_size=100000,
            # trainer args
            log_dir='/tmp/rl/log',
            total_train_steps=50000,
            print_freq=1000,
            save_freq=10000,
            ):
        self.d = d
        self._build()

    def _build(self):
        logging.info('device: {}.'.format(self._device))

        self._repr_fn = self._build_model_haiku()
        self._optimizer = optax.adam(0.001)
        self.train_step_jax = jax.jit(self.train_step_jax)

        self._replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
                max_size=self._replay_buffer_size)
        self._global_step = 0
        self._train_info = collections.OrderedDict()

    def _build_model_haiku(self):
        def lap_net(obs):
            network = hk.Sequential(
                [nets.MLP([256, 256, 256, self.d])])
            return network(obs.astype(np.float32))
        return hk.without_apply_rng(hk.transform(lap_net))

    def _random_policy_fn(self, state):
        return self._action_spec.sample(), None

    def _get_obs_batch(self, steps):
        obs_batch = [self._obs_prepro(s.step.time_step.observation)
                for s in steps]
        return np.stack(obs_batch, axis=0)

    def _get_rew_batch(self, steps):
        rew_batch = [s.step.time_step.reward
                for s in steps]
        return np.stack(rew_batch, axis=0)

    def _get_train_batch_jax(self,):
        s1, s2 = self._replay_buffer.sample_pairs(
                batch_size=self._batch_size//32,
                discount=self._discount,
                )
        s_neg = self._replay_buffer.sample_steps(self._batch_size)
        s_neg_2 = self._replay_buffer.sample_steps(self._batch_size)
        s1_pos, s2_pos, s_neg, s_neg_2 = map(self._get_obs_batch, [s1, s2, s_neg, s_neg_2])
        reward_s1, reward_s2 = map(self._get_rew_batch,[s1, s2])
        batch = Data(s1_pos, s2_pos, reward_s2, s_neg, s_neg_2)
        return batch

    def _loss(self, params, batch,):
        s1_repr = self._repr_fn.apply(params, batch.s1)
        s2_repr = self._repr_fn.apply(params, batch.s2)
        s_neg_repr = self._repr_fn.apply(params, batch.s_neg)
        s_neg_2_repr = self._repr_fn.apply(params, batch.s_neg_2)
        loss, loss_positive, loss_negative = generalized_graph_drawing_loss_haiku(s1_repr, s2_repr,
                                            s_neg_repr, s_neg_2_repr, batch.reward)
        return loss, (loss, loss_positive, loss_negative)

    def train_step_jax(self, params, opt_state, batch):
        # self._loss(params, batch)
        dloss_dtheta, aux = jax.grad(self._loss, has_aux=True)(params, batch)
        updates, opt_state = self._optimizer.update(dloss_dtheta, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, aux

    def train(self):
        saver_dir = self._log_dir
        if not os.path.exists(saver_dir):
            os.makedirs(saver_dir)
        actor = actors.StepActor(self._env_factory)

        # start actors, collect trajectories from random actions
        logging.info('Start collecting samples.')
        timer = timer_tools.Timer()

        # collect initial transitions
        total_n_steps = 0
        collect_batch = 10000
        while total_n_steps < self._n_samples:
            n_steps = min(collect_batch, 
                    self._n_samples - total_n_steps)
            steps = actor.get_steps(n_steps, self._random_policy_fn)
            self._replay_buffer.add_steps(steps)
            total_n_steps += n_steps
            logging.info('({}/{}) steps collected.'
                .format(total_n_steps, self._n_samples))
        time_cost = timer.time_cost()
        logging.info('Data collection finished, time cost: {}s'
            .format(time_cost))

        seed = 1337
        rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
        sample_input = self._get_train_batch_jax()
        params = self._repr_fn.init(next(rng), sample_input.s1)
        opt_state = self._optimizer.init(params)


        # learning begins
        timer.set_step(0)
        for step in range(self._total_train_steps):
            assert step == self._global_step

            batch = self._get_train_batch_jax()
            params, opt_state, losses = self.train_step_jax(params, opt_state, batch)
            self._global_step += 1
            self._train_info['loss_total'] = np.array([jax.device_get(losses[0])])[0]
            self._train_info['loss_pos'] = np.array([jax.device_get(losses[1])])[0]
            self._train_info['loss_neg'] = np.array([jax.device_get(losses[2])])[0]

            # save
            if (step + 1) % self._save_freq == 0:
                saver_path = os.path.join(saver_dir, 'model-{}.pkl'.format(step+1))
                self.save_ckpt(saver_path, params)
            # print info
            if step == 0 or (step + 1) % self._print_freq == 0:
                steps_per_sec = timer.steps_per_sec(step)
                logging.info('Training steps per second: {:.4g}.'.format(steps_per_sec))
                self._print_train_info()

        saver_path = os.path.join(saver_dir, 'model.pkl')
        self.save_ckpt(saver_path, params)
        time_cost = timer.time_cost()
        logging.info('Training finished, time cost {:.4g}s.'.format(time_cost))


    def _print_train_info(self):
        # import pdb;pdb.set_trace()
        summary_str = summary_tools.get_summary_str(
                step=self._global_step, info=self._train_info)
        logging.info(summary_str)

    def save_ckpt(self, filepath, params):
        # import pdb;pdb.set_trace()
        numpy_params = jax.device_get(params)
        with open(filepath, 'wb') as file:
            pickle.dump(numpy_params, file)
        # jnp.save(params, filepath)
        # jax.device_get(params)
        # torch.save(self._repr_fn.state_dict(), filepath)
