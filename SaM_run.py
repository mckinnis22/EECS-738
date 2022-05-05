#from pyexpat import model
#from spinup import ppo_pytorch as ppo
import numpy as np
import gym
import torch
import time
import os.path as osp
from os.path import exists
from torch.optim import Adam
import spinup.algos.pytorch.dqn.core as core
from spinup.algos.pytorch.dqn.core import DQNetwork as DQN
from spinup.algos.pytorch.dqn.dqn import *
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


model_path = '/home/jef/spinningup/spinup/Results/pyt_save/'
model_name = 'TryCatchingThisOneMotherFucker.pt'

env = gym.make('SaM-minnow-v0')

seed = 0

# Random seed
seed += 10000 * proc_id()
torch.manual_seed(seed)
np.random.seed(seed)

#Here is what is going on. The dqn.py is made like the spinningup algorithms in that its stupid simple to
#run with normal examples. However, as soon as issues arose with the custom environment, Mozammal told me
#to break up their code line by line to make debugging a possibility.

#DQN setup
dqnetwork=DQN
steps_per_epoch=1000
epochs=500
replay_size=int(1e6)
gamma=0.99
min_replay_history=20000
epsilon_decay_period=250000
epsilon_train=0.01
epsilon_eval=0.001
lr=1e-3
max_ep_len=500
update_period=4
target_update_period=8000
batch_size=100
logger_kwargs = dict(output_dir='Results', exp_name='testing1')
save_freq=20

ac_kwargs = dict()

#setting up the logger that creates the output data table
logger = EpochLogger(**logger_kwargs)

#describing the observation and action spaces as inputs to the NN
obs_dim = env.observation_space.n #not sure about this, but it should desccribe the 25*50 tensor made by our play grid
act_dim = 1 #believe this just says you get to pick one action
ac_kwargs["action_space"] = env.action_space

#setting up the NN parts
main = dqnetwork(in_features=obs_dim, **ac_kwargs)
target = dqnetwork(in_features=obs_dim, **ac_kwargs)

#Load Model
if exists(osp.join(model_path,model_name)):
    main.load_state_dict(torch.load(osp.join(model_path,model_name))) #this is how you load a model


replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

var_counts = tuple(core.count_vars(module) for module in [main.q, main])

#telling the nn the parameters in a different format
value_params = main.q.parameters()

#basic optimizer setup
value_optimizer = torch.optim.Adam(value_params, lr=lr)

# Initializing targets to match main variables
target.load_state_dict(main.state_dict())
#this is a leftover piece from DQN, ig you could possibly use a different test environment to get real spooky
test_env = env

#next two functions are taken directly
def get_action(o, epsilon):
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

def test_agent(n=10):
    for _ in range(n):
        o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
        while not (d or (ep_len == max_ep_len)):
            # epsilon_eval used when evaluating the agent
            o, r, d, _ = test_env.step(get_action(o, epsilon_eval))
            ep_ret += r
            ep_len += 1
        logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

#setup to save model after each epoch
# logger.setup_pytorch_saver(main)
#pre run setup, takes time, resets the environment class
start_time = time.time()
o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
total_steps = steps_per_epoch * epochs

#none of this should have changed from the dqn.py
for t in range(total_steps):
    # if t == 5982:
    # print(t)
    main.eval()

    # the epsilon value used for exploration during training
    epsilon = core.linearly_decaying_epsilon(
        epsilon_decay_period, t, min_replay_history, epsilon_train
    )
    a = get_action(o, epsilon)

    # Step the env
    o2, r, d, _ = env.step(a)
    ep_ret += r
    ep_len += 1

    # Ignore the "done" signal if it comes from hitting the time
    # horizon (that is, when it's an artificial terminal signal
    # that isn't based on the agent's state)
    d = False if ep_len == max_ep_len else d

    # Store experience to replay buffer
    replay_buffer.store(o, a, r, o2, d)

    # Super critical, easy to overlook step: make sure to update
    # most recent observation!
    o = o2

    if d or (ep_len == max_ep_len):
        logger.store(EpRet=ep_ret, EpLen=ep_len)
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    # train at the rate of update_period if enough training steps have been run
    if replay_buffer.size > min_replay_history and t % update_period == 0:
        main.train()
        batch = replay_buffer.sample_batch(batch_size)
        (obs1, obs2, acts, rews, done) = (
            torch.Tensor(batch["obs1"]),
            torch.Tensor(batch["obs2"]),
            torch.Tensor(batch["acts"]),
            torch.Tensor(batch["rews"]),
            torch.Tensor(batch["done"]),
        )
        q_pi = main(obs1).gather(1, acts.long()).squeeze()
        q_pi_targ, _ = target(obs2).max(1)

        # Bellman backup for Q function
        backup = (rews + gamma * (1 - done) * q_pi_targ).detach()

        # DQN loss
        value_loss = F.smooth_l1_loss(q_pi, backup)

        # Q-learning update
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
        logger.store(LossQ=value_loss.item(), QVals=q_pi.data.numpy())

    # syncs weights from online to target network
    if t % target_update_period == 0:
        target.load_state_dict(main.state_dict())

    # End of epoch wrap-up
    if replay_buffer.size > min_replay_history and t % steps_per_epoch == 0:
        epoch = t // steps_per_epoch

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            
            #this is how you you save the model, i like to name them as a funciton of UNIX time
            # model_name = 'model_%s.pt' % time.time()
            model_name = 'epoch%s.pt' % epoch
            torch.save(main.state_dict(), osp.join(model_path,model_name))
        # Test the performance of the deterministic version of the agent.
        test_agent()

        # Log info about epoch
        logger.log_tabular("EpRet", with_min_and_max=True)
        logger.log_tabular("TestEpRet", with_min_and_max=True)
        logger.log_tabular("EpLen", average_only=True)
        logger.log_tabular("TestEpLen", average_only=True)
        logger.log_tabular("TotalEnvInteracts", t)
        logger.log_tabular("QVals", with_min_and_max=True)
        logger.log_tabular("LossQ", average_only=True)
        logger.log_tabular("Time", time.time() - start_time)
        logger.log_tabular("Epoch", epoch)
        logger.dump_tabular()
        # print('Current Epoch: %s' % epoch)
