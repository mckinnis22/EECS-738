from pyexpat import model
from matplotlib.pyplot import grid
import numpy as np
import gym
import torch
import os.path as osp
import spinup.algos.pytorch.dqn.core as core
from spinup.algos.pytorch.dqn.core import DQNetwork as DQN
from spinup.algos.pytorch.dqn.dqn import *

model_path = '/home/aaron/spinningup/spinup/Results/pyt_save'

dqnetwork=DQN
ac_kwargs = dict()

grid_state = np.zeros((25,50))

env_shark = gym.make('SaM-v8')
env_minnow = gym.make('SaM-v9')

# grid_state = np.zeros((25,50))
obs_dim_shark = env_shark.observation_space.n #not sure about this, but it should desccribe the 25*50 tensor made by our play grid
act_dim = 1 #believe this just says you get to pick one action
ac_kwargs["action_space"] = env_shark.action_space

#setting up the NN parts
main_shark = dqnetwork(in_features=obs_dim_shark, **ac_kwargs)
main_shark.load_state_dict(torch.load(osp.join(model_path,'model_working_8moves.pt')))

obs_dim_minnow = env_minnow.observation_space.n
act_dim = 1
ac_kwargs["action_space"] = env_minnow.action_space



main_minnow = dqnetwork(in_features=obs_dim_minnow, **ac_kwargs)
main_minnow.load_state_dict(torch.load(osp.join(model_path,'minnow.pt')))

test_steps = 200

o_shark, r_shark, d_shark, ep_ret_shark, ep_len_shark = env_shark.reset(), 0, False, 0, 0

grid_state = env_shark.share_grid()
env_minnow.receive_grid(grid_state)

o_minnow, r_minnow, d_minnow, ep_ret_minnow, ep_len_minnow = env_minnow.reset(), 0, False, 0, 0

grid_state = env_shark.share_grid()

epsilon_eval=0.001


def get_action_shark(o,epsilon):
    """Select an action from the set of available actions.
    Chooses an action randomly with probability epsilon otherwise
    act greedily according to the current Q-value estimates.
    """
    if np.random.random() <= epsilon:
        return env_shark.action_space.sample()
    else:
        q_values = main_shark(torch.Tensor(o.reshape(1, -1)))
        # return the action with highest Q-value for this observation
        return torch.argmax(q_values, dim=1).item()

def get_action_minnow(o,epsilon):
    """Select an action from the set of available actions.
    Chooses an action randomly with probability epsilon otherwise
    act greedily according to the current Q-value estimates.
    """
    if np.random.random() <= epsilon:
        return env_minnow.action_space.sample()
    else:
        q_values = main_minnow(torch.Tensor(o.reshape(1, -1)))
        # return the action with highest Q-value for this observation
        return torch.argmax(q_values, dim=1).item()


for i in range(test_steps):
    main_shark.eval()
    main_minnow.eval()

    # grid_state[10,38] = 1
    
    

    if not i % 3:
        a_shark = get_action_shark(o_shark,epsilon_eval)

        o_shark, r_shark, d_shark, _ = env_shark.step(a_shark)

        grid_state = env_shark.share_grid()


    env_minnow.receive_grid(grid_state)

    a_minnow = get_action_minnow(o_minnow,epsilon_eval)

    o_minnow, r_minnow, d_minnow, _ = env_minnow.step(a_minnow)

    grid_state = env_minnow.share_grid()

    env_shark.receive_grid(grid_state)


    env_shark.render()

    if d_shark + d_minnow >= 1:
        break