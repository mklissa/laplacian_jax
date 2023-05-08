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
from . import gt_laplacian

Data = collections.namedtuple("Data", "s1 s2 reward s_neg_x s_neg_y s_neg_z")


rate = 2.
dimensionality = 20
coeff_vector = jnp.arange(dimensionality, 0, -1)
# coeff_vector = 1. / jnp.arange(1, dimensionality + 1)
# coeff_vector = 1. / rate ** jnp.arange(dimensionality)
# coeff_vector = jnp.ones(dimensionality)
# coeff_vector = coeff_vector.at[0].set(3)
coeff_vector = jnp.concatenate((coeff_vector, jnp.zeros(1)))
print(coeff_vector[:-1])
coeff = 0.3

def kernel_loss(pos_rep_i, pos_rep_j):
    cov = jnp.outer(pos_rep_i, pos_rep_j)
    diag = jnp.diagonal(cov)
    sg_cov = jnp.outer(jax.lax.stop_gradient(pos_rep_i), pos_rep_j)**2
    non_diag = (sg_cov - jnp.diagonal(sg_cov)) / 2
    return diag.sum() - coeff * non_diag.sum()

def neg_loss_fn(neg_rep_x, neg_rep_y, neg_rep_z):
    loss = 0
    n_dim = neg_rep_x.shape[0]
    for dim in range(n_dim, 0, -1):
        coeff = coeff_vector[dim-1] - coeff_vector[dim]

        x_norm = jnp.sqrt(jnp.dot(neg_rep_x[:dim], neg_rep_x[:dim]))
        y_norm = jnp.sqrt(jnp.dot(neg_rep_y[:dim], neg_rep_y[:dim]))

        delta = 1.0
        dot_product = jnp.dot(neg_rep_x[:dim], neg_rep_y[:dim])
        absolute = jnp.abs(dot_product)
        quadratic = jnp.minimum(absolute, delta)
        linear = absolute - quadratic
        orthogonality_loss = 0.5 * quadratic ** 2 + delta * linear

        loss += coeff * ( absolute - jnp.log(1 + x_norm / n_dim)  - jnp.log(1 + y_norm / n_dim)  )
            #- jnp.dot(neg_rep_z[:dim], neg_rep_z[:dim]) / n_dim )
        # # max_norm = jnp.maximum(x_norm, y_norm)
        # loss += coeff * ( jnp.dot(neg_rep_x[:dim], neg_rep_y[:dim]) ) ** 2#  - x_norm ** 2 / n_dim - y_norm ** 2 / n_dim )
    # loss = jnp.minimum(1, loss)
    # loss = jnp.maximum(-1, jnp.minimum(1, loss))
    return loss
    # return jnp.exp(loss)

def pos_loss_fn(pos_rep_i, pos_rep_j):
    norm_i = jnp.sqrt(jnp.dot(pos_rep_i, pos_rep_i))
    norm_j = jnp.sqrt(jnp.dot(pos_rep_j, pos_rep_j))
    return 1. - jnp.dot(pos_rep_i, pos_rep_j) / (norm_i * norm_j)

def generalized_graph_drawing_loss_haiku(pos_rep_i, pos_rep_j, neg_rep_x, neg_rep_y, neg_rep_z, alpha=1.0, beta=2.0):
    # kernel_vmap = jax.vmap(kernel_loss)
    # loss = kernel_vmap(pos_rep_i, pos_rep_j).mean()
    # pos_loss = 0.
    # neg_loss = 0.
    pos_loss = ((pos_rep_i - pos_rep_j)**2).dot(coeff_vector[:dimensionality]).mean()
    pos_loss = jnp.minimum(1, pos_loss)
    # pos_loss_vmap = jax.vmap(pos_loss_fn)
    # pos_loss  = pos_loss_vmap(pos_rep_i, pos_rep_j).mean()

    neg_loss_vmap = jax.vmap(neg_loss_fn)
    # neg_loss  = jnp.log(neg_loss_vmap(neg_rep_x, neg_rep_y, neg_rep_z).sum())
    # neg_loss  = neg_loss_vmap(neg_rep_x, neg_rep_y, neg_rep_z).max()
    neg_loss  = neg_loss_vmap(neg_rep_x, neg_rep_y, neg_rep_z).mean()
    # neg_loss = jnp.maximum(-2, jnp.minimum(2, neg_loss))

    loss = alpha * pos_loss + beta * neg_loss

    return loss, pos_loss, neg_loss


def _build_model_haiku(d):
    def lap_net(obs):
        activation = jax.nn.relu
        network = hk.Sequential([
            hk.Linear(256),
            activation,
            hk.Linear(256),
            activation,
            hk.Linear(256),
            activation,
            hk.Linear(d),
            # jax.nn.elu,
            ])
        return network(obs.astype(np.float32))
    return hk.without_apply_rng(hk.transform(lap_net))


class LapReprLearner:

    @py_tools.store_args
    def __init__(self,
            d,
            max_distance,
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

        self._repr_fn = _build_model_haiku(self._d)
        self._optimizer = optax.adam(0.001)
        self.train_step_jax = jax.jit(self.train_step_jax)

        self._replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
                max_size=self._replay_buffer_size)
        self._global_step = 0
        self._train_info = collections.OrderedDict()

    def _random_policy_fn(self, state):
        action = self._action_spec.sample()
        return action, None

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
                # batch_size=self._batch_size,
                batch_size=self._batch_size,
                discount=self._discount,
                )
        s_neg_x = self._replay_buffer.sample_steps(self._batch_size)
        s_neg_y = self._replay_buffer.sample_steps(self._batch_size)
        s_neg_z = self._replay_buffer.sample_steps(self._batch_size)
        s1_pos, s2_pos, s_neg_x, s_neg_y, s_neg_z = map(self._get_obs_batch, [s1, s2, s_neg_x, s_neg_y, s_neg_z])
        reward_s1, reward_s2 = map(self._get_rew_batch,[s1, s2])
        batch = Data(s1_pos, s2_pos, reward_s2, s_neg_x, s_neg_y, s_neg_z)
        return batch

    def _loss(self, params, batch, alpha, beta):
        s1_repr = self._repr_fn.apply(params, batch.s1)
        s2_repr = self._repr_fn.apply(params, batch.s2)
        s_neg_x_repr = self._repr_fn.apply(params, batch.s_neg_x)
        s_neg_y_repr = self._repr_fn.apply(params, batch.s_neg_y)
        s_neg_z_repr = self._repr_fn.apply(params, batch.s_neg_z)
        loss, loss_positive, loss_negative = generalized_graph_drawing_loss_haiku(s1_repr, s2_repr,
                                            s_neg_x_repr, s_neg_y_repr, s_neg_z_repr, alpha=alpha, beta=beta)
        return loss, (loss, loss_positive, loss_negative)

    def train_step_jax(self, params, opt_state, batch, alpha, beta):
        # import pdb;pdb.set_trace()
        _, aux = self._loss(params, batch, alpha, beta)
        dloss_dtheta, aux = jax.grad(self._loss, has_aux=True)(params, batch, alpha, beta)
        updates, opt_state = self._optimizer.update(dloss_dtheta, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, aux

    def calculate_simgt(self, params, states_batch):
        laprep = self._repr_fn.apply(params, states_batch)
        laprep_normalized = laprep / jnp.linalg.norm(laprep, axis=0)

        all_sim_gt = laprep_normalized[:, :].T @ jnp.array(self.d_small_eigvecs[:, :])
        # sim_gt = jnp.abs(jnp.diagonal(all_sim_gt)).mean()
        sim_gt = jnp.mean(jnp.max(jnp.abs(all_sim_gt), axis=1))
        sim_gt = np.array([sim_gt])[0]

        # all_sim_gt = np.array(all_sim_gt)
        # sim_gt = []
        # for i in range(len(all_sim_gt)):
        #     x = max(i-2, 0)
        #     y = min(i+3, len(all_sim_gt))
        #     cur_sim = np.max(np.abs(all_sim_gt)[i, x:y])
        #     sim_gt.append(cur_sim)
        # sim_gt = np.array(sim_gt).mean()

        self._train_info['sim_gt'] = sim_gt
        return sim_gt

    def train(self):
        saver_dir = self._log_dir
        if not os.path.exists(saver_dir):
            os.makedirs(saver_dir)
        actor = actors.StepActor(self._env_factory)

        # Calculate Ground-Truth Eigenvectors
        env = self._env_factory()
        if hasattr(env, 'admissible_scalar_stickers'):
            # eigenvectors = np.load(f'./rl_lap/data/eigenvectors_distance{self._max_distance}.npy')
            eigenvectors = np.zeros((env.num_states, self._d))
            self.d_small_eigvecs = eigenvectors[:, :self._d]
            states_batch = env.admissible_obs
            # states_batch = np.eye(env.num_states)
        else:
            states, r_states = gt_laplacian.get_all_states(env)
            eigenvectors, eigenvalues, P = gt_laplacian.get_exact_laplacian(states, r_states)
            self.d_small_eigvecs = d_small_eigvecs = eigenvectors[:, :self._d]

            n_states = env.task.maze.n_states
            pos_batch = env.task.maze.all_empty_grids()
            obs_batch = [env.task.pos_to_obs(pos) for pos in pos_batch]
            def obs_prepro(obs):
                return obs.agent.position
            obs_batch = [env.task.pos_to_obs(pos_batch[i]) for i in range(n_states)]
            states_batch = np.array([obs_prepro(obs) for obs in obs_batch])

        # start actors, collect trajectories from random actions
        logging.info('Start collecting samples.')
        timer = timer_tools.Timer()

        # collect initial transitions
        total_n_steps = 0
        collect_batch = 10_000
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
        final_alpha = 1.0
        final_beta = 0.3
        period_alpha = self._total_train_steps / 2.
        period_beta = self._total_train_steps / 2.
        for step in range(self._total_train_steps):
            assert step == self._global_step

            alpha = final_alpha * min((step/period_alpha), 1)
            # alpha = final_alpha
            beta = final_beta * min((step/period_beta), 1)
            # beta = final_beta

            batch = self._get_train_batch_jax()
            params, opt_state, losses = self.train_step_jax(params, opt_state, batch, alpha=alpha, beta=beta)
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
                self.calculate_simgt(params, states_batch)
                steps_per_sec = timer.steps_per_sec(step)
                logging.info('Steps per second: {:.4g}.'.format(steps_per_sec))
                self._print_train_info()

        saver_path = os.path.join(saver_dir, 'model.pkl')
        self.save_ckpt(saver_path, params)
        time_cost = timer.time_cost()
        logging.info('Training finished, time cost {:.4g}s.'.format(time_cost))
        sim_gt = self.calculate_simgt(params, states_batch)
        logging.info(f'Final SimGT: {sim_gt:.4f}')

    def _print_train_info(self):
        summary_str = summary_tools.get_summary_str(
                step=self._global_step, info=self._train_info)
        logging.info(summary_str)

    def save_ckpt(self, filepath, params):
        numpy_params = jax.device_get(params)
        with open(filepath, 'wb') as file:
            pickle.dump(numpy_params, file)
