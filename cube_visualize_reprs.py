"""Visualize learned representation."""
import os
import argparse
import importlib


import networkx as nx
import numpy as np
import torch
import matplotlib
# matplotlib.use('Agg')
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


parser = argparse.ArgumentParser()
parser.add_argument('--log_base_dir', type=str, 
        default=os.path.join(os.getcwd(), 'log'))
parser.add_argument('--log_sub_dir', type=str, 
        default='laprepr/cube/torch')
parser.add_argument('--output_sub_dir', type=str, 
        default='visualize_reprs/torch')
# parser.add_argument('--output_sub_dir', type=str, 
#         default='visualize_reprs/eigenvectors')
parser.add_argument('--config_dir', type=str, default='rl_lap.configs')
parser.add_argument('--config_file', 
        type=str, default='laprepr_config_gridworld')


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
    model = learner_args.model_cfg.model_factory()
    model.to(device=device)
    ckpt_path = os.path.join(log_dir, 'model.ckpt')
    model.load_state_dict(torch.load(ckpt_path))

    distance = 2
    env = learner_args.env_factory()
    eigenvectors = np.load(f'./rl_lap/data/eigenvectors_distance{distance}.npy')
    d_small_eigvecs = eigenvectors[:, :]
    d_small_eigvecs = torch_tools.to_tensor(d_small_eigvecs, device)
    states_batch = env.admissible_stickers
    # states_batch = np.eye(env.num_states)


    # get representations from loaded model
    states_torch = torch_tools.to_tensor(states_batch, device)
    lap_rep = model(states_torch)
    states_reprs = lap_rep.detach().cpu().numpy()

    eigen_id = 1
    indices = env.admissible_states
    values = states_reprs[:, eigen_id]
    # values = d_small_eigvecs[:, eigen_id]
    print(values)

    def to_str(number):
      return f'{number:.2}'
    labels = dict(enumerate(list(map(to_str, values))))

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


    plt.figure(figsize=(8,8))
    pos = nx.nx_pydot.pydot_layout(G, prog="twopi")

    nx.draw_networkx(G, pos=pos, with_labels=0,
      cmap='Reds', node_size=50, font_size=20, font_weight='bold', node_color=values, labels=labels) 
        
    plt.show()


if __name__ == '__main__':
    main()

