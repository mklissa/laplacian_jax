from ..agent import laprepr
from ..envs.gridworld import gridworld_envs
from . import networks

from ..rubiks222 import Rubiks2x2x2

class Config(laprepr.LapReprConfig):

    def _set_default_flags(self):
        super()._set_default_flags()
        flags = self._flags
        flags.d = 10
        flags.n_samples = 50_000
        flags.batch_size = 256
        flags.discount = 0.0
        flags.w_neg = 2.0
        flags.c_neg = 1.0
        flags.reg_neg = 0.0
        flags.replay_buffer_size = 50_000
        flags.opt_args.name = 'Adam'
        flags.opt_args.lr = 0.0001
        # train
        flags.log_dir = '/tmp/rl_laprepr/log'
        flags.total_train_steps = 20_000
        flags.print_freq = 1000
        flags.save_freq = 10000

        flags.max_distance = 9

    def _obs_prepro(self, obs):
        if 'cube' in self._flags.env_id:
            return obs
        else:
            return obs.agent.position

    def _env_factory(self):
        if 'cube' in self._flags.env_id:
            env = Rubiks2x2x2(max_distance=self._flags.max_distance,
                                random_restarts=True)
        else:
            env =  gridworld_envs.make(self._flags.env_id)
        return env

    def _model_factory(self):
        return networks.ReprNetMLP(
                self._obs_shape, n_layers=3, n_units=256,
                d=self._flags.d)
