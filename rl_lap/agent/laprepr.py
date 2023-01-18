import os
import logging
import collections

import numpy as np
import torch
from torch import optim

from . import episodic_replay_buffer
from ..envs import actors
from ..tools import py_tools
from ..tools import torch_tools
from ..tools import flag_tools
from ..tools import summary_tools
from ..tools import timer_tools

def generalized_graph_drawing_loss(pos_rep_i, pos_rep_j, neg_rep, neg_rep_2, reward, delta, beta, ggd=True):

    # Retreive the dimensions
    bz, n_dim = neg_rep.shape[:2]
    # all_reward = reward[:, None] + 1.
    # import pdb;pdb.set_trace()
    pos_loss, neg_loss = 0, 0
    for dim in range(n_dim, 0, -1):
        # Loss for positive pairs
        # pos_loss += ( all_reward + 0.99 * pos_rep_j[:, :dim] - pos_rep_i[:, :dim] ).pow(2).sum(dim=-1).mean()
        pos_loss += (pos_rep_i[:, :dim] - pos_rep_j[:, :dim]).pow(2).sum(dim=-1).mean()

        # all_dot_products = torch.matmul(neg_rep[:, :dim], neg_rep_2[:, :dim].T)
        # loss = torch.square(all_dot_products.diagonal()) / n_dim - neg_rep[:, :dim].pow(2).sum(dim=-1) / n_dim - neg_rep_2[:, :dim].pow(2).sum(dim=-1) / n_dim + 1./ n_dim
        # neg_loss += loss.mean()
        # # Loss for negative pairs
        inprods = neg_rep[:, :dim] @ neg_rep[:, :dim].T
        norms = torch.diagonal(inprods, 0)
        part1 = (inprods.pow(2).sum() - norms.pow(2).sum()) / (bz * (bz - 1))
        part2 = -2 * delta * norms.mean() / n_dim
        part3 = delta * delta / n_dim
        neg_loss += part1 + part2 + part3

        # Break the loop if not using generalized graph drawing objective
        if not ggd:
            break

    # Total loss
    loss = pos_loss  + beta * neg_loss 
    return loss, pos_loss, neg_loss


def l2_dist(x1, x2):
    return (x1 - x2).pow(2).sum(-1)


def pos_loss(x1, x2):
    return l2_dist(x1, x2).mean()


def neg_loss(x, c=1.0, reg=0.0):
    """
    x: n * d.
    sample based approximation for
    (E[x x^T] - c * I / d)^2
        = E[(x^T y)^2] - 2c E[x^T x] / d + c^2 / d
    #
    An optional regularization of
    reg * E[(x^T x - c)^2] / n
        = reg * E[(x^T x)^2 - 2c x^T x + c^2] / n
    for reg in [0, 1]
    """

    # import pdb;pdb.set_trace()
    n = x.shape[0]
    d = x.shape[1]
    inprods = x @ x.T
    norms = inprods[torch.arange(n), torch.arange(n)]
    part1 = inprods.pow(2).sum() - norms.pow(2).sum()
    part1 = part1 / ((n - 1) * n)
    part2 = - 2 * c * norms.mean() / d
    part3 = c * c / d
    # regularization
    if reg > 0.0:
        reg_part1 = norms.pow(2).mean()
        reg_part2 = - 2 * c * norms.mean()
        reg_part3 = c * c
        reg_part = (reg_part1 + reg_part2 + reg_part3) / n
    else:
        reg_part = 0.0
    return part1 + part2 + part3 + reg * reg_part


class LapReprLearner:

    @py_tools.store_args
    def __init__(self,
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
        self._build()

    def _build(self):
        if self._device is None:
            self._device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
        logging.info('device: {}.'.format(self._device))
        self._build_model()
        self._build_optimizer()
        self._replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
                max_size=self._replay_buffer_size)
        self._global_step = 0
        self._train_info = collections.OrderedDict()

    def _build_model(self):
        cfg = self._model_cfg
        self._repr_fn = cfg.model_factory()
        self._repr_fn.to(device=self._device)

    def _build_optimizer(self):
        cfg = self._optimizer_cfg
        self._optimizer = cfg.optimizer_factory(self._repr_fn.parameters())

    def _build_loss(self, batch):
        s1 = batch.s1
        s2 = batch.s2
        s_neg = batch.s_neg
        s_neg_2 = batch.s_neg_2
        s1_repr = self._repr_fn(s1)
        s2_repr = self._repr_fn(s2)
        s_neg_repr = self._repr_fn(s_neg)
        s_neg_2_repr = self._repr_fn(s_neg_2)

        loss, loss_positive, loss_negative = generalized_graph_drawing_loss(s1_repr, s2_repr,
                                            s_neg_repr, s_neg_2_repr, batch.reward,
                                            delta=1.0, beta=self._w_neg, ggd=True)

        info = self._train_info
        info['loss_pos'] = loss_positive.item()
        info['loss_neg'] = loss_negative.item()
        info['loss_total'] = loss.item()
        return loss
    
    def _random_policy_fn(self, state):
        return self._action_spec.sample(), None

    def _get_obs_batch(self, steps):
        obs_batch = [self._obs_prepro(s.step.time_step.observation)
                for s in steps]
        return np.stack(obs_batch, axis=0)


    def _get_obs_batch(self, steps):
        obs_batch = [self._obs_prepro(s.step.time_step.observation)
                for s in steps]
        return np.stack(obs_batch, axis=0)


    def _get_rew_batch(self, steps):
        rew_batch = [s.step.time_step.reward
                for s in steps]
        return np.stack(rew_batch, axis=0)

    def _tensor(self, x):
        return torch_tools.to_tensor(x, self._device)

    def _get_train_batch(self):
        s1, s2 = self._replay_buffer.sample_pairs(
                batch_size=self._batch_size,
                discount=self._discount,
                )
        s_neg = self._replay_buffer.sample_steps(self._batch_size)
        s_neg_2 = self._replay_buffer.sample_steps(self._batch_size)
        s1_pos, s2_pos, s_neg, s_neg_2 = map(self._get_obs_batch, [s1, s2, s_neg, s_neg_2])
        batch = flag_tools.Flags()
        reward_s1, reward_s2 = map(self._get_rew_batch,[s1, s2])
        batch.s1 = self._tensor(s1_pos)
        batch.s2 = self._tensor(s2_pos)
        batch.reward = self._tensor(reward_s2)
        batch.s_neg = self._tensor(s_neg)
        batch.s_neg_2 = self._tensor(s_neg_2)
        return batch

    def _train_step(self):
        train_batch = self._get_train_batch()
        loss = self._build_loss(train_batch)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._global_step += 1

    def _print_train_info(self):
        summary_str = summary_tools.get_summary_str(
                step=self._global_step, info=self._train_info)
        logging.info(summary_str)

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

        # learning begins
        timer.set_step(0)
        for step in range(self._total_train_steps):
            assert step == self._global_step
            self._train_step()
            # save
            if (step + 1) % self._save_freq == 0:
                saver_path = os.path.join(saver_dir, 
                        'model-{}.ckpt'.format(step+1))
                self.save_ckpt(saver_path)
            # print info
            if step == 0 or (step + 1) % self._print_freq == 0:
                steps_per_sec = timer.steps_per_sec(step)
                logging.info('Training steps per second: {:.4g}.'
                        .format(steps_per_sec))
                self._print_train_info()
        saver_path = os.path.join(saver_dir, 'model.ckpt')
        self.save_ckpt(saver_path)
        time_cost = timer.time_cost()
        logging.info('Training finished, time cost {:.4g}s.'.format(time_cost))

    def save_ckpt(self, filepath):
        torch.save(self._repr_fn.state_dict(), filepath)


class LapReprConfig(flag_tools.ConfigBase):

    def _set_default_flags(self):
        flags = self._flags
        flags.device = None
        flags.env_id = None
        # agent
        flags.d = 20
        flags.n_samples = 10000
        flags.batch_size = 128
        flags.discount = 0.9
        flags.w_neg = 1.0
        flags.c_neg = 1.0
        flags.reg_neg = 0.0
        flags.replay_buffer_size = 100000
        flags.opt_args = flag_tools.Flags(name='Adam', lr=0.001)
        # train
        flags.log_dir = './log/generalized'
        flags.total_train_steps = 50000
        flags.print_freq = 1000
        flags.save_freq = 10000

    def _build(self):
        self._build_env()
        self._build_model()
        self._build_optimizer()
        self._build_args()


    def _obs_prepro(self, obs):
        return obs

    def _env_factory(self):
        raise NotImplementedError

    def _model_factory(self):
        raise NotImplementedError

    def _optimizer_factory(self, parameters):
        opt = getattr(optim, self._flags.opt_args.name)
        opt_fn = opt(parameters, lr=self._flags.opt_args.lr)
        return opt_fn

    def _build_env(self):
        dummy_env = self._env_factory()
        dummy_time_step = dummy_env.reset()
        self._action_spec = dummy_env.action_spec
        self._obs_shape = list(self._obs_prepro(
            dummy_time_step.observation).shape)

    def _build_model(self):
        self._model_cfg = flag_tools.Flags(
                model_factory=self._model_factory)

    def _build_optimizer(self):
        self._optimizer_cfg = flag_tools.Flags(
                optimizer_factory=self._optimizer_factory)

    def _build_args(self):
        args = flag_tools.Flags()
        args.device = self._flags.device
        # env args
        args.action_spec = self._action_spec
        args.obs_shape = self._obs_shape
        args.obs_prepro = self._obs_prepro
        args.env_factory = self._env_factory
        # learner args
        args.model_cfg = self._model_cfg
        args.optimizer_cfg = self._optimizer_cfg
        args.n_samples = self._flags.n_samples
        args.batch_size = self._flags.batch_size
        args.discount = self._flags.discount
        args.w_neg = self._flags.w_neg
        args.c_neg = self._flags.c_neg
        args.reg_neg = self._flags.reg_neg
        args.replay_buffer_size = self._flags.replay_buffer_size
        # training args
        args.log_dir = self._flags.log_dir
        args.total_train_steps = self._flags.total_train_steps
        args.print_freq = self._flags.print_freq
        args.save_freq = self._flags.save_freq
        self._args = args

    @property
    def args(self):
        return vars(self._args)

    @property
    def args_as_flags(self):
        return self._args

