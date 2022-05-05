from dis import dis
import numpy as np
import gym
from gym import error, spaces
from gym import utils
import os.path as osp


import matplotlib.pyplot as plt
from matplotlib import colors

class SaMEnv_2(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SaMEnv_2,self).__init__()

        self.grid_state = np.zeros((25,50),dtype=float)
        self.number_shark = 2

        self.action_space  = spaces.Discrete(8)
        self.observation_space = spaces.Discrete(2)
        self.dist_count = 0
        self.t = 0
        self.agent_pos = np.empty(2*self.number_shark,)
        self.previous_pos = np.empty(2*self.number_shark,)
        self.eaten_minnows = 0

    def reset(self):
        #this function creates a random game space for the sharks and minnows game
        #2 vertical walls of random length and location
        #6 static minnows
        #1 hungry shark
        #static end goal (not used)
        z = np.zeros((25,50))
        self.t = 0

        # ------------ minnows --------------- 
        minnow_x = np.random.random_integers(low = 1, high = 24,size = 100)
        minnow_y = np.random.random_integers(low = 1, high = 40,size = 100)

        z[minnow_x,minnow_y] = 2

        #need same logic here
        shark_x = np.random.random_integers(low = 2, high = 23,size = 2)
        shark_y = np.random.random_integers(low = 10, high = 48,size = 2)

        x = np.nonzero(z[shark_x,shark_y])

        shark_y[x] = shark_y[x] - 1
        shark_x[x] = shark_x[x] - 1
        try:
            z[shark_x,shark_y] = 3
        except:
            z[15,49] = 3

        self.agent_pos = np.array([shark_x,shark_y]).T

        
        # z[np.random.random_integers(low = 2, high = 23,size = 1),49] = 4
        self.grid_state = z


        rel_x = np.zeros(self.number_shark)
        rel_y = np.zeros(self.number_shark)

        return np.array([rel_x,rel_y]).flatten()

    def step(self,action):
        #this is the function where you need to propogate the agent as well as assign the reward for actions
        #Currently the reward is punishing movement around the board and slightly rewarding movement towards the
        #lower bound, once the lower bound is reach a large reward is given and the done flag is set to 1
        # reward = -1
        
        done = 0
        reward = -1 * np.ones(self.number_shark,)
        
        self.t += 1

        self.previous_pos = self.agent_pos
        # a = self.number_shark * [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]]
        action_space = np.array([[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]])
        
        minnow_action_space = np.array([[1,0],[0,1],[-1,0],[0,-1]])
        
        
        self.agent_pos = self.agent_pos + action_space[action]
        
        unq, count = np.unique(self.agent_pos,axis = 0, return_counts=True)

        a  = unq[count >1]
        if len(a) > 0:
            for x in np.arange(0,np.shape(a)[0]):


                idx = np.where(np.all(self.agent_pos == a[x,:], axis = 1))

                self.agent_pos[idx[0][1],:] = self.previous_pos[idx[0][1],:] 
        
        #stuff for later
        tmp2 = np.where(self.grid_state ==2) #finds minnow locations
        minnow = np.vstack((tmp2[0],tmp2[1])).T
        
        # if self.t % 2:
        #     self.grid_state[tmp2[0],tmp2[1]] = 5
        #     random_act = np.random.random_integers(low = 0,high = 3, size = len(tmp2[0]))
        #     minnow[:,0] = minnow[:,0] + minnow_action_space[random_act][:,0]
        #     minnow[:,0] = np.clip(minnow[:,0],0,24)
        #     minnow[:,1] = minnow[:,1] + minnow_action_space[random_act][:,1]
        #     minnow[:,1] = np.clip(minnow[:,1],0,49)
        #     self.grid_state[minnow[:,0],minnow[:,1]] = 2


        rel_x = np.zeros(self.number_shark)
        rel_y = np.zeros(self.number_shark)
        min_list= []


        for y in np.arange(0,self.number_shark):

            
            list_dist = []
            for x in range(0,int(np.size(minnow)/2)):
                dist = np.sqrt((self.agent_pos[y,0] -minnow[x,0])**2 + (self.agent_pos[y,1] -minnow[x,1])**2)
                
                if dist < 1.75:
                    self.grid_state[minnow[x][0],minnow[x][1]] = 7
                    reward[y] += 100
                    self.eaten_minnows += 1
                else:
                    list_dist.append(dist)
            if list_dist:
                min_index = list_dist.index(min(list_dist))# check for shortest distance always returns lowest index
                min_list.append(min_index)
                if y >= 1 and len(list_dist) > 1:
                    if not min_list[y] - min_list[y-1] :
                        list_dist.pop(min_list[y])
                        min_index = list_dist.index(min(list_dist))


                
                rel_x_prev = minnow[min_index,0]- self.previous_pos[y,0]
                rel_y_prev = minnow[min_index,1] - self.previous_pos[y,1]

                rel_x[y] = minnow[min_index,0]- self.agent_pos[y,0]
                rel_y[y] = minnow[min_index,1] - self.agent_pos[y,1]


                if abs(rel_x[y]) < abs(rel_x_prev):
                    reward[y] += 2
                elif rel_x[y] == rel_x_prev:
                    reward[y] = reward[y]
                else:
                    reward[y] -=2

                
                if abs(rel_y[y]) < abs(rel_y_prev):
                    reward[y] += 2
                elif rel_y[y] == rel_y_prev:
                    reward[y] = reward[y]
                else:
                    reward[y] -=2


            if self.agent_pos[y,0] == -1 or self.agent_pos[y,0] == 25:
                self.agent_pos[y,:] = self.previous_pos[y,:]
                # reward -= 10
                # done = 1
            if self.agent_pos[y,1] == -1 or self.agent_pos[y,1] == 50:
                self.agent_pos[y,:] = self.previous_pos[y,:]
                # reward -= 10
                # done = 1
        

        self.grid_state[self.previous_pos[:,0],self.previous_pos[:,1]] = 0
        self.grid_state[self.agent_pos[:,0],self.agent_pos[:,1]] = 3

        if not minnow.size > 0:
            done = 1
            reward[:,] += 50

        info = {}


        return np.array([rel_x,rel_y]).T.flatten(), reward, done, info

    def get_agent_pos(self):
        return self.agent_pos

    def share_grid(self):

        return self.grid_state

    def receive_grid(self,grid_state):

        self.grid_state = grid_state

        return

    def render(self, minnow_success, mode='human'):
        #call this function to render the board
        path = '/home/aaron/spinningup/spinup/Results/Figures'
        name = 'fig_%s' %self.t

        #DO NOT CALL IN TRAINING
        #you will absolutely freeze your computer
        cmap = colors.ListedColormap(['blue', 'yellow','#c2c2c2','red','green','white','#ff9494','black'])
        bounds=[0,1,2,3,4,5,6,7,8]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        # fig = plt.figure()

        im = plt.matshow(np.flip(self.grid_state,axis = 0), interpolation='nearest', cmap = cmap,norm=norm)
        title_txt = "Sharks: {}  Minnows: {}".format(self.eaten_minnows,minnow_success)
        plt.title (title_txt)
        plt.savefig(osp.join(path,name))
        plt.show(block=False)

    def update_obs(self):
        tmp2 = np.where(self.grid_state ==2) #finds minnow locations
        minnow = np.vstack((tmp2[0],tmp2[1])).T
        
        # if self.t % 2:
        #     self.grid_state[tmp2[0],tmp2[1]] = 5
        #     random_act = np.random.random_integers(low = 0,high = 3, size = len(tmp2[0]))
        #     minnow[:,0] = minnow[:,0] + minnow_action_space[random_act][:,0]
        #     minnow[:,0] = np.clip(minnow[:,0],0,24)
        #     minnow[:,1] = minnow[:,1] + minnow_action_space[random_act][:,1]
        #     minnow[:,1] = np.clip(minnow[:,1],0,49)
        #     self.grid_state[minnow[:,0],minnow[:,1]] = 2

        done = 0
        rel_x = np.zeros(self.number_shark)
        rel_y = np.zeros(self.number_shark)
        min_list= []


        for y in np.arange(0,self.number_shark):

            
            list_dist = []
            for x in range(0,int(np.size(minnow)/2)):
                dist = np.sqrt((self.agent_pos[y,0] -minnow[x,0])**2 + (self.agent_pos[y,1] -minnow[x,1])**2)
                
                if dist < 1.75:
                    self.grid_state[minnow[x][0],minnow[x][1]] = 7
                    self.eaten_minnows += 1
                else:
                    list_dist.append(dist)
            if list_dist:
                min_index = list_dist.index(min(list_dist))# check for shortest distance always returns lowest index
                min_list.append(min_index)
                if y >= 1 and len(list_dist) > 1:
                    if not min_list[y] - min_list[y-1] :
                        list_dist.pop(min_list[y])
                        min_index = list_dist.index(min(list_dist))


                
                rel_x_prev = minnow[min_index,0]- self.previous_pos[y,0]
                rel_y_prev = minnow[min_index,1] - self.previous_pos[y,1]

                rel_x[y] = minnow[min_index,0]- self.agent_pos[y,0]
                rel_y[y] = minnow[min_index,1] - self.agent_pos[y,1]


        if self.agent_pos[y,0] == -1 or self.agent_pos[y,0] == 25:
            self.agent_pos[y,:] = self.previous_pos[y,:]
            # reward -= 10
            # done = 1
        if self.agent_pos[y,1] == -1 or self.agent_pos[y,1] == 50:
            self.agent_pos[y,:] = self.previous_pos[y,:]
            # reward -= 10
            # done = 1
      

        self.grid_state[self.previous_pos[:,0],self.previous_pos[:,1]] = 0
        self.grid_state[self.agent_pos[:,0],self.agent_pos[:,1]] = 3

        if not minnow.size > 0:
            done = 1
            

        return np.array([rel_x,rel_y]).T.flatten(),done

    def close(self):
        #not necessary if only running on cpu
        pass
        
        
