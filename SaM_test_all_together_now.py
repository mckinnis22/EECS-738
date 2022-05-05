from pyexpat import model
from turtle import position
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

number_shark = 2
env_shark = gym.make('SaM-v2')
env_minnow = gym.make('SaM-v9')

grid_state = np.zeros((25,50))
obs_dim = env_shark.observation_space.n #not sure about this, but it should desccribe the 25*50 tensor made by our play grid
act_dim = 1 #believe this just says you get to pick one action
ac_kwargs["action_space"] = env_shark.action_space

#setting up the NN parts
shark1 = dqnetwork(in_features=obs_dim, **ac_kwargs)
shark1.load_state_dict(torch.load(osp.join(model_path,'model_working_8moves.pt')))

# shark2 = dqnetwork(in_features=obs_dim, **ac_kwargs)
# shark2.load_state_dict(torch.load(osp.join(model_path,'model_working_8moves.pt')))

obs_dim = env_minnow.observation_space.n #not sure about this, but it should desccribe the 25*50 tensor made by our play grid
act_dim = 1 #believe this just says you get to pick one action
ac_kwargs["action_space"] = env_minnow.action_space

minnow1 = dqnetwork(in_features=obs_dim, **ac_kwargs)
minnow1.load_state_dict(torch.load(osp.join(model_path,'MinnowvSharkSameSpeed.pt')))







test_steps = 200

o_shark, r_shark, d_shark, ep_ret_shark, ep_len_shark = env_shark.reset(), 0, False, 0, 0
grid_state = env_shark.share_grid()
env_minnow.receive_grid(grid_state)



o_minnow, r_minnow, d_minnow, ep_ret_minnow, ep_len_minnow = env_minnow.reset(), 0, False, 0, 0
epsilon_eval=0.001


grid_state = env_shark.share_grid()
env_minnow.receive_grid(grid_state)

# def get_action(o,epsilon):
#     """Select an action from the set of available actions.
#     Chooses an action randomly with probability epsilon otherwise
#     act greedily according to the current Q-value estimates.
#     """
#     if np.random.random() <= epsilon:
#         return env.action_space.sample()
#     else:
#         q_values = main(torch.Tensor(o.reshape(1, -1)))
#         # return the action with highest Q-value for this observation
#         return torch.argmax(q_values, dim=1).item()

for i in range(test_steps):
    shark1.eval()

    minnow1.eval()
    

    a= []
    # a1 = torch.argmax(shark1(torch.Tensor(o_shark[0:2].reshape(1, -1))), dim=1).item()
    # a2 = torch.argmax(shark1(torch.Tensor(o_shark[2:].reshape(1, -1))), dim=1).item()
    for x in np.arange(0,len(o_shark),2):
        a.append(torch.argmax(shark1(torch.Tensor(o_shark[x:x+2].reshape(1, -1))), dim=1).item())

    o_shark, r_shark, d_shark, _ = env_shark.step(a)

    grid_state = env_shark.share_grid() #this grid will have the positions of the dead minnows

    if d_shark == 1:
        env_shark.render(win)
        break

    env_minnow.receive_grid(grid_state)

    o_minnow = env_minnow.update_obs()

    # env_shark.render()

    a = []

    for x in np.arange(0,len(o_minnow),4):
        a.append(torch.argmax(minnow1(torch.Tensor(o_minnow[x:x+4].reshape(1, -1))), dim=1).item())

    o_minnow, r_minnow, d_minnow, _ = env_minnow.step(a)

    grid_state = env_minnow.share_grid()
    env_shark.receive_grid(grid_state)

    o_shark, d_shark = env_shark.update_obs()
    
    win = env_minnow.share_win()
    env_shark.render(win)

    if d_shark == 1:
        
        break