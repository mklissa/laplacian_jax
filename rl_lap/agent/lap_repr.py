import logging
import collections
import torch
from torch import optim

from . import episodic_replay_buffer
from ..envs import actors
from ..tools import py_tools
from ..tools import torch_tools
from ..tools import flag_tools
from ..tools import summary_tools


def l2_dist(x):
  x = tf.reshape(x, [x.shape.as_list()[0], -1])
  return tf.reduce_sum(tf.square(x), axis=-1)

def pos_loss(x1, x2):
  '''
  x1, x2: batch_size * repr_dim.
  Return: a loss that positively correlates with the distances between x1 and x2.
  '''
  return tf.reduce_mean(l2_dist(x1 - x2))


def neg_loss(x, c=1.0):
  '''
  Return: a loss for negative sampling.
  '''
  n = x.shape.as_list()[0]
  dot_prods = tf.matmul(x, tf.transpose(x))
  loss = tf.reduce_sum(tf.square(dot_prods / n - c * tf.eye(n))) # should be 1 or n^2
  return loss

def neg_loss_unbiased(x1, x2, c=1.0):
  '''
  x1, x2: n*d
  return E[(x'y)^2] - 2cE[x'x] + c^2 * d
  '''
  d = x1.shape.as_list()[1]
  part1 = tf.reduce_mean(tf.square(tf.matmul(x1, tf.transpose(x2))))
  part2 = - c * (tf.reduce_mean(tf.reduce_sum(tf.square(x1), axis=-1))
                 + tf.reduce_mean(tf.reduce_sum(tf.square(x2), axis=-1)))
  part3 = c * c * d
  loss = part1 + part2 + part3
  return loss



def neg_loss_sigmoid(x, c=1.0):
  '''
  c \in [0, 1]
  '''
  n = x.shape.as_list()[0]
  dot_prods = tf.matmul(x, tf.transpose(x))
  loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=dot_prods, labels=c*tf.eye(n))) / n # should be 1 or n^2
  return loss



class LapReprLearner:

    @pytools.store_args
    def __init__(self,
            # env args
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
            replay_buffer_size=10000,
            # trainer args
            log_dir='/tmp/rl/log',
            total_train_steps=50000,
            print_freq=1000,
            save_freq=10000,
            # pytorch
            device=None,
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
        self._train_info = collections.ordereddict()

    def _build_model(self):
        pass

    def _build_optimizer(self):
        cfg = self._optimizer_cfg
        opt_fn = getattr(optim, cfg.name)
        self._optimizer = opt_fn(
                self._repr_fn.parameters(),
                lr=cfg.lr,
                )

    def _build_loss(self, batch):
        raise NotImplementedError
    
    def _random_policy_fn(self, state):
        return self._action_spec.sample(), None

    def _get_obs_batch(self, steps):
        # each step is a tuple of (time_step, action, context)
        obs_batch = [self._obs_prepro(ts[0].observation) for ts in steps]
        return np.stack(obs_batch, axis=0)

    def _tensor(self, x):
        return torch_tools.to_tensor(x, self._device)

    def _get_train_batch(self):
        s1, s2 = self._replay_buffer.sample_pairs(
                batch_size=self._batch_size,
                discount=self._discount,
                )
        s_neg = self._replay_buffer.sample_steps(self._batch_size)
        s1, s2, s_neg = map(self._get_obs_batch, [s1, s2, s_neg])
        batch = flag_tools.Flags()
        batch.s1 = self._tensor(s1)
        batch.s2 = self._tensor(s2)
        batch.s_neg = self._tensor(s_neg)
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
        evaluator = evaluation.BasicEvaluator(self._test_env_factory)
        result_path = os.path.join(saver_dir, 'result.csv')

        # start actors, collect trajectories from random actions
        logging.info('Start collecting transitions.')
        start_time = time.time()
        # collect initial transitions
        total_n_steps = 0
        collect_batch = 10000
        while total_n_steps < self._replay_buffer_init:
            n_steps = min(collect_batch, 
                    self._replay_buffer_init - total_n_steps)
            steps = actor.get_steps(n_steps, self._random_policy_fn)
            self._replay_buffer.add_steps(steps)
            total_n_steps += n_steps
            logging.info('({}/{}) steps collected.'
                .format(total_n_steps, self._replay_buffer_init))
        time_cost = time.time() - start_time
        logging.info('Replay buffer initialization finished, time cost: {}s'
            .format(time_cost))
        # learning begins
        start_time = time.time()
        test_results = []
        for step in range(self._total_train_steps):
            assert step == self._global_step
            self._train_step()
            # update replay memory:
            if (step + 1) % self._replay_update_freq == 0:
                steps = actor.get_steps(self._replay_update_num,
                        self._train_policy_fn)
            self._replay_buffer.add_steps(steps)
            # save
            if (step + 1) % self._save_freq == 0:
                saver_path = os.path.join(saver_dir, 
                        'agent-{}.ckpt'.format(step+1))
                self.save_ckpt(saver_path)
            # print info
            if step == 0 or (step + 1) % self._print_freq == 0:
                time_cost = time.time() - start_time
                logging.info('Training steps per second: {:.4g}.'
                        .format(self._print_freq/time_cost))
                self._print_train_info()
                start_time = time.time()
            # test
            if step == 0 or (step + 1) % self._test_freq == 0:
                tst = time.time()
                test_result = evaluator.run_test(self._n_test_episodes,
                        self._test_policy_fn)
                edt = time.time()
                test_results.append(
                        [step+1] + list(test_result) + [edt-tst])
                self._print_test_info(test_results)
        saver_path = os.path.join(saver_dir, 'agent.ckpt')
        self.save_ckpt(saver_path)
        test_results = np.array(test_results)
        np.savetxt(result_path, test_results, fmt='%.4g', delimiter=',')

    def save_ckpt(self, filepath):
        torch.save(self._repr_fn.state_dict(), filepath)


class BaseModules(object):

    def __init__(self, modules, obs_shape, action_spec):
        self._modules = modules
        self._obs_shape = obs_shape
        self._action_spec = action_spec

    def build(self, model='learn', device=torch.device('cpu')):
        # should return an object that contains (e.g.):
        # vars_save, vars_train, vars_sync
        raise NotImplementedError


  def _build_learner(self):

    # set up tf graph for training
    self._train_holders = self._build_train_holders()
    # instantiate networks
    with tf.variable_scope(self._scope_name):
      with tf.variable_scope('repr_net'):
        self._repr_net = self._repr_module()

    repr_s1, repr_s2, repr_s_neg = map(
        self._repr_net,
        [self._train_holders.s1, self._train_holders.s2,
         self._train_holders.s_neg]
    )
    self._loss_pos = pos_loss(repr_s1, repr_s2)
    self._loss_neg = neg_loss(repr_s_neg, c=self._neg_loss_c)

    self._loss = self._loss_pos + self._w_neg * self._loss_neg

    # optimization
    opt = self._optimizer(learning_rate=self._learning_rate)
    self._train_op = opt.minimize(self._loss)
    # saver
    self._var_list = self._repr_net.get_variables()
    self._model_saver = tf.train.Saver(var_list=self._var_list)

  def _build_train_holders(self):
    holders = EmptyClass()
    holders.s1 = tf.placeholder(tf.float32, [self._pos_batch_size]+self._obs_shape)
    holders.s2 = tf.placeholder(tf.float32, [self._pos_batch_size]+self._obs_shape)
    holders.s_neg = tf.placeholder(tf.float32, [self._neg_batch_size]+self._obs_shape)
    #holders.a = tf.placeholder(tf.int32, [self._batch_size])
    #holders.r = tf.placeholder(tf.float32, [self._batch_size])
    #holders.dsc = tf.placeholder(tf.float32, [self._batch_size])  # discount
    return holders

  def _get_obs_repr(self, obs):
    """overload for specific use"""
    return self._obs_to_repr(obs)

  @property
  def obs_shape(self):
    return self._obs_shape

  @property
  def obs_to_repr(self):
    return self._obs_to_repr

  @property
  def repr_net(self):
    return self._repr_net

  @property
  def repr_module(self):
    return self._repr_module


  def _get_obs_batch(self, time_steps):
    # each time step is a pair of (action, time_step)
    obs_batch = [self._get_obs_repr(ts[1].observation) for ts in time_steps]
    return np.stack(obs_batch, axis=0)


  #def _get_action_batch(self, time_steps):
  #  action_batch = [ts[0] for ts in time_steps]
  #  return np.stack(action_batch, axis=0)


  def _get_train_batch(self, replay_memory):
    if self._sample_max_range > 1:
      ts1, ts2 = replay_memory.sample_positive(
          batch_size=self._pos_batch_size,
          discount=self._sample_discount,
          max_range=self._sample_max_range,
      )
    else:
      ts1, ts2 = replay_memory.sample_transitions(self._pos_batch_size)
    ts_neg = replay_memory.sample_steps(
        batch_size=self._neg_batch_size,
    )
    #a = self._get_action_batch(ts1)
    s1, s2, s_neg = map(self._get_obs_batch, [ts1, ts2, ts_neg])
    # compute reward and discount
    #r, dsc = self._get_r_dsc_batch(ts2)
    holders = self._train_holders
    feed_dict = {
        holders.s1: s1,
        holders.s2: s2,
        holders.s_neg: s_neg,
        }
    return feed_dict

  def _get_batch_dict(self, feed_dict):
    vals_dict = {}
    for kw, holder in vars(self._train_holders).items():
      vals_dict[kw] = feed_dict[holder]
    return vals_dict

  def train_step(self, sess, replay_memory, step, print_freq):
    #sess = self._session
    feed_dict = self._get_train_batch(replay_memory)
    loss, loss_pos, loss_neg, _ = sess.run(
        [self._loss, self._loss_pos, self._loss_neg, self._train_op],
        feed_dict=feed_dict)
    # print info
    if step == 0 or (step + 1) % print_freq == 0:
      logging.info(('Step {}:  loss {:.4g}, loss_pos {:.4g}, loss_neg {:.4g}.'
                   ).format(step+1, loss, loss_pos, loss_neg))
      #print(('Step {}:  loss {:.4g}, loss_pos {:.4g}, loss_neg {:.4g}.'
      #      ).format(step+1, loss, loss_pos, loss_neg))
    batch = self._get_batch_dict(feed_dict)
    self._global_step += 1
    return batch

  def save_model(self, sess, path):
      self._model_saver.save(sess, path)

  def load_model(self, sess, path):
      self._model_saver.restore(sess, path)


class W2VUnbiasedLearner(W2VLearner):

  def _build_learner(self):

    # set up tf graph for training
    self._train_holders = self._build_train_holders()
    # instantiate networks
    with tf.variable_scope(self._scope_name):
      with tf.variable_scope('repr_net'):
        self._repr_net = self._repr_module()

    repr_s1, repr_s2, repr_s1_neg, repr_s2_neg = map(
        self._repr_net,
        [self._train_holders.s1, self._train_holders.s2,
         self._train_holders.s1_neg, self._train_holders.s2_neg]
    )
    self._loss_pos = pos_loss(repr_s1, repr_s2)
    self._loss_neg = neg_loss_unbiased(repr_s1_neg, repr_s2_neg, c=self._neg_loss_c)

    self._loss = self._loss_pos + self._w_neg * self._loss_neg

    # optimization
    opt = self._optimizer(learning_rate=self._learning_rate)
    self._train_op = opt.minimize(self._loss)
    # saver
    self._var_list = self._repr_net.get_variables()
    self._model_saver = tf.train.Saver(var_list=self._var_list)

  def _build_train_holders(self):
    holders = EmptyClass()
    holders.s1 = tf.placeholder(tf.float32, [self._pos_batch_size]+self._obs_shape)
    holders.s2 = tf.placeholder(tf.float32, [self._pos_batch_size]+self._obs_shape)
    holders.s1_neg = tf.placeholder(tf.float32, [self._neg_batch_size]+self._obs_shape)
    holders.s2_neg = tf.placeholder(tf.float32, [self._neg_batch_size]+self._obs_shape)
    return holders

  def _get_train_batch(self, replay_memory):
    ts1, ts2 = replay_memory.sample_transitions(
        batch_size=self._pos_batch_size,
    )
    ts1_neg = replay_memory.sample_steps(
        batch_size=self._neg_batch_size,
    )
    ts2_neg = replay_memory.sample_steps(
        batch_size=self._neg_batch_size,
    )

    #a = self._get_action_batch(ts1)
    s1, s2, s1_neg, s2_neg = map(
        self._get_obs_batch, [ts1, ts2, ts1_neg, ts2_neg])
    # compute reward and discount
    #r, dsc = self._get_r_dsc_batch(ts2)
    holders = self._train_holders
    feed_dict = {
        holders.s1: s1,
        holders.s2: s2,
        holders.s1_neg: s1_neg,
        holders.s2_neg: s2_neg,
        }
    return feed_dict

