from pyexpat import model
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

env = gym.make('SaM-v8')


# grid_state = np.zeros((25,50))
obs_dim = env_shark.observation_space.n #not sure about this, but it should desccribe the 25*50 tensor made by our play grid
act_dim = 1 #believe this just says you get to pick one action
ac_kwargs["action_space"] = env_shark.action_space




#setting up the NN parts
main = dqnetwork(in_features=obs_dim, **ac_kwargs)
main.load_state_dict(torch.load(osp.join(model_path,'model_working_8moves.pt')))


test_steps = 200

o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0


epsilon_eval=0.001


def get_action(o,epsilon):
    """Select an action from the set of available actions.
    Chooses an action randomly with probability epsilon otherwise
    act greedily according to the current Q-value estimates.
    """
    if np.random.random() <= epsilon:
        return env.action_space.sample()
    else:
        q_values = main(torch.Tensor(o.reshape(1, -1)))
        # return the action with highest Q-value for this observation
        return torch.argmax(q_values, dim=1).item()




for i in range(test_steps):
    main.eval()

    # grid_state[10,38] = 1
    

    a = get_action(o,epsilon_eval)

    o, r, d, _ = env.step(a)



    env.render()

    if d == 1:
        break
