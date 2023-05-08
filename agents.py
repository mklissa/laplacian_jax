import numpy as np

class Agent():
  def __init__(self, num_states, num_actions, batch_size, lr, counts=0.0, option_set=None):
    self.params = np.zeros((num_states, num_actions))
    self.state_visitation = np.zeros(num_states,)
    self.num_states_visited = 0.
    self.num_states = num_states
    self.num_actions = num_actions
    self.batch_size = batch_size
    self.lr = lr
    self.option_set = option_set
    self.option = None
    self.termination = True
    self.epsilon = 0.1
    self.mu = 0.8 if self.option_set is not None else  0.0
    self.duration = 10.
    self.counts = counts

  def act(self, state):
    if self.state_visitation[state] == 0.:
      self.num_states_visited += 1.
    self.state_visitation[state] += 1.
    self.termination = np.random.rand() < 1 / self.duration \
                        or self.termination

    if self.termination:
      if np.random.rand() < self.epsilon:
        if np.random.rand() < self.mu:
          self.option = np.random.randint(len(self.option_set))
          self.termination = False
          action = self.option_set[self.option][state]
        else:
          self.option = None
          self.termination = True
          action = np.random.randint(self.num_actions)
      else:
        action = self.greedy_act(state)
    else:
      action = self.option_set[self.option][state]

    return action

  def greedy_act(self, state):
    return np.argmax(self.params[state])

  def train_step(self, params, buffer):
    batch = buffer.sample(self.batch_size)
    q_tm1 = self.params[batch.s_tm1, batch.a_tm1]
    q_t = self.params[batch.s_t]
    reward = batch.r_t + self.counts * 1. / np.sqrt(self.state_visitation[batch.s_tm1])

    td_errors = reward + batch.discount_t * np.max(q_t, axis=1) - q_tm1
    self.params[batch.s_tm1, batch.a_tm1] += self.lr * td_errors


  def online_train(self, params, batch):
    q_tm1 = self.params[batch.s_tm1, batch.a_tm1]
    q_t = self.params[batch.s_t]
    reward = batch.r_t + self.counts * 1. / np.srqt(self.state_visitation[batch.s_tm1])

    td_errors = reward + batch.discount_t * np.max(q_t) - q_tm1
    self.params[batch.s_tm1, batch.a_tm1] += self.lr * td_errors
