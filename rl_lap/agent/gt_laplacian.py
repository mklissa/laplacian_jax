import torch


def get_all_states(env):
    # states = [
    #     [x, y]
    #     for x in range(env.task.maze._width)
    #     for y in range(env.task.maze._height)
    #     if env.task.maze._maze[x, y] == ' '
    # ]
    states = env.task.maze.all_empty_grids()
    states = torch.tensor(states)
    r_states = torch.full((env.task.maze._width, env.task.maze._height), fill_value=-1, dtype=torch.long)
    r_states[states[:, 0], states[:, 1]] = torch.arange(0, len(states))
    return states, r_states


def get_exact_laplacian(states, r_states, n_actions=4):
    # Build adjacency matrix
    A = torch.zeros((len(states), len(states)))
    states_map = r_states > -1
    cur_pos = r_states[states_map]
    left_pos = r_states[states_map.roll(shifts=-1, dims=1)]
    left_pos[left_pos == -1] = cur_pos[left_pos == -1]
    A[cur_pos, left_pos] = 1
    right_pos = r_states[states_map.roll(shifts=1, dims=1)]
    right_pos[right_pos == -1] = cur_pos[right_pos == -1]
    A[cur_pos, right_pos] = 1
    up_pos = r_states[states_map.roll(shifts=-1, dims=0)]
    up_pos[up_pos == -1] = cur_pos[up_pos == -1]
    A[cur_pos, up_pos] = 1
    down_pos = r_states[states_map.roll(shifts=1, dims=0)]
    down_pos[down_pos == -1] = cur_pos[down_pos == -1]
    A[cur_pos, down_pos] = 1

    # Build transition matrix
    P = A / n_actions
    P[range(len(states)), range(len(states))] += 1 - P.sum(axis=0)

    # Compute graph Laplacian
    D = A.sum(axis=-1)
    L = torch.diag(D) - A

    # Eigendecompose the Laplacian
    eigenvalues, eigenvectors = torch.linalg.eig(L)
    idx = torch.real(eigenvalues).argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return torch.real(eigenvectors), torch.real(eigenvalues), P