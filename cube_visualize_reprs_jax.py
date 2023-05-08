"""Visualize learned representation."""
import os
import argparse
import importlib
import pickle

import networkx as nx
import numpy as np
import torch
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

from rl_lap.agent import laprepr
from rl_lap.tools import flag_tools
from rl_lap.tools import torch_tools
from rl_lap.agent import gt_laplacian
from rl_lap.agent import laprepr_jax


parser = argparse.ArgumentParser()
parser.add_argument('--log_base_dir', type=str, 
        default=os.path.join(os.getcwd(), 'log'))
parser.add_argument('--log_sub_dir', type=str, 
        default='laprepr/cube/jax')
parser.add_argument('--output_sub_dir', type=str, 
        default='visualize_reprs/jax')
# parser.add_argument('--output_sub_dir', type=str, 
#         default='visualize_reprs/eigenvectors')
parser.add_argument('--config_dir', type=str, default='rl_lap.configs')
parser.add_argument('--config_file', 
        type=str, default='laprepr_config_gridworld')
parser.add_argument('--iteration', 
        type=int, default=50_000)


FLAGS = parser.parse_args()


def get_config_cls():
    config_module = importlib.import_module(
            FLAGS.config_dir+'.'+FLAGS.config_file)
    config_cls = config_module.Config
    return config_cls

def main():
    # setup log directories
    log_dir = os.path.join(FLAGS.log_base_dir, FLAGS.log_sub_dir)
    output_dir = os.path.join(FLAGS.log_base_dir, FLAGS.output_sub_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load config
    flags = flag_tools.load_flags(log_dir)
    cfg_cls = get_config_cls()
    cfg = cfg_cls(flags)
    learner_args = cfg.args_as_flags
    device = learner_args.device

    # load model from checkpoint
    filepath = log_dir + f'/model-{FLAGS.iteration}.pkl'
    with open(filepath, 'rb') as file:
        params = pickle.load(file)
    model = laprepr_jax._build_model_haiku(cfg.flags.d)

    # distance = 3
    # eigenvectors = np.load(f'./rl_lap/data/eigenvectors_distance{distance}.npy')
    # d_small_eigvecs = eigenvectors[:, :]

    # get representations from loaded model
    env = learner_args.env_factory()
    states_batch = env.admissible_obs
    states_reprs = model.apply(params, states_batch)
    indices = env.admissible_states

    G=nx.Graph()
    all_states = {}
    for i, ind in enumerate(indices):
      G.add_node(i, at=env.distances[ind])
      all_states[ind] = i

    for i, ind in enumerate(indices):
      neighbours = env.action_matrices[:,ind]
      for nei in neighbours:
        if nei in all_states:
          G.add_edge(i, all_states[nei]) 

    for eigen in range(min(10,cfg.flags.d)):

        values = states_reprs[:, eigen]
        # values = d_small_eigvecs[:, eigen]
        print(values)

        plt.figure(figsize=(8,8))
        pos = nx.nx_pydot.pydot_layout(G, prog="twopi")

        nx.draw_networkx(G, pos=pos, with_labels=0,
          cmap='Reds', node_size=50, font_size=20, font_weight='bold', node_color=values)

        figfile = os.path.join(output_dir, f'{flags.env_id}_eigen{eigen}.png')
        plt.savefig(figfile, bbox_inches='tight')
        plt.clf()


if __name__ == '__main__':
    main()

