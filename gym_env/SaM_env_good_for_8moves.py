from dis import dis
import numpy as np
import gym
from gym import error, spaces
from gym import utils
import os.path as osp
import matplotlib.pyplot as plt
from matplotlib import colors

class SaMEnv_8(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SaMEnv_8,self).__init__()

        self.grid_state = np.zeros((25,50),dtype=float)
        self.checked_space = np.zeros((25,50),dtype=float)
        #self.action_space = spaces.Discrete(4)
        self.action_space  = spaces.Discrete(8)
        self.observation_space = spaces.Discrete(2)
        self.dist_count = 0
        self.t = 0
        #defines the grid we are playing in and the possible values for each
        # self.observation_space = spaces.Box(low = 0, high = 4, shape=(25,50),dtype=np.uint8) 
        #self.observation_space = spaces.Discrete(1250)
        self.agent_pos = np.empty(2,)
        self.previous_pos = np.empty(2,)


    def reset(self):
        #this function creates a random game space for the sharks and minnows game
        #2 vertical walls of random length and location
        #6 static minnows
        #1 hungry shark
        #static end goal (not used)
        z = np.zeros((25,50))
        self.t = 0
        # # self.dist_count = 0
        # # self.checked_space = np.zeros((25,50),dtype=float)
        # # self.grid_state = np.zeros((25,50),dtype=float)
        # # -------- walls --------------------- #
        # #creates two wall starters
        # wall_seeds = np.random.random_integers(low = 10, high = 40,size = 2)

        # #makes sure they cannot be too close

        # if abs(wall_seeds[1] - wall_seeds[0]) < 3:
        #     wall_seeds[1] = wall_seeds[1] + 5

        # #decides whehterh should start from the top or bottom
        # tob = np.random.random_integers(low = 0, high = 1,size = 2)

        # #decides on length
        # wall_length = np.random.random_integers(low = 2, high = 10,size = 2)

        # for i in range(0,len(wall_seeds)):
        #     #if tob (top or bottom) = 0 start from the top, if 1 the bottom
        #     if tob[i] == 0:
        #         z[0:wall_length[i],wall_seeds[i]] = 1
        #     if tob[i] == 1:
        #         z[25-wall_length[i]:25,wall_seeds[i]] = 1

        # ------------ static minnows --------------- 
        minnow_x = np.random.random_integers(low = 1, high = 24,size = 1)
        minnow_y = np.random.random_integers(low = 1, high = 40,size = 1)
        # minnow_x = 12
        # minnow_y = 7

        #need a logic to not start a minnow on a wall

        # x = np.nonzero(z[minnow_x,minnow_y])

        # minnow_y[x] = minnow_y[x] - 1

        z[minnow_x,minnow_y] = 2

        #need same logic here
        shark_x = np.random.random_integers(low = 2, high = 23,size = 1)
        shark_y = np.random.random_integers(low = 30, high = 48,size = 1)

        x = np.nonzero(z[shark_x,shark_y])

        shark_y[x] = shark_y[x] - 1
        shark_x[x] = shark_x[x] - 1
        try:
            z[shark_x,shark_y] = 3
        except:
            z[15,49] = 3

        self.agent_pos = np.array([shark_x,shark_y]).flatten()

        
        # z[21:24,49] = 4
        self.grid_state = z

        tmp2 = np.where(self.grid_state ==2) #finds minnow locations
        minnow = np.vstack((tmp2[0],tmp2[1])).T

        dist = 0
        list_dist = []
        for x in range(0,int(np.size(minnow)/2)):
            dist = np.sqrt((self.agent_pos[0] -minnow[x,0])**2 + (self.agent_pos[1] -minnow[x,1])**2)
            list_dist.append(dist)
        min_index = list_dist.index(min(list_dist))


        rel_x = minnow_x[min_index] - self.agent_pos[0]
        rel_y = minnow_y[min_index] - self.agent_pos[1]

        return np.array([rel_x,rel_y]).flatten()

    def step(self,action):
        #this is the function where you need to propogate the agent as well as assign the reward for actions
        #Currently the reward is punishing movement around the board and slightly rewarding movement towards the
        #lower bound, once the lower bound is reach a large reward is given and the done flag is set to 1
        # reward = -1
        done = 0
        reward = -1
        self.t += 1

        self.previous_pos = self.agent_pos

        action_space = np.array([[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]])
        minnow_action_space = np.array([[1,0],[0,1],[-1,0],[0,-1]])
        

        self.agent_pos = np.squeeze(self.agent_pos) + action_space[action]
        
        

        #following mozammal's advice attepting to train the agent to only go straight 
        #if agent going down give reward, for this iteration I did not care about intersections with minnows/walls
        # if self.previous_pos[0] > self.agent_pos[0]:
        #     reward +=2
        #     #if agent went straight down give additional reward
        #     if self.previous_pos[1] == self.agent_pos[1]:
        #         reward +=1
        # #end if agent hits each of map        
        # if self.agent_pos[0] == -1 or self.agent_pos[0] == 24:
        #     self.agent_pos = self.previous_pos
        #     reward += 60
        #     done = 1
        # if self.agent_pos[1] == -1 or self.agent_pos[1] == 49:
        #     self.agent_pos = self.previous_pos
        #     reward -= 5
        #     done = 1

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

        # distance check on minnow and eat it if you there (works)
        dist = 0
        list_dist = []
        for x in range(0,int(np.size(minnow)/2)):
            dist = np.sqrt((self.agent_pos[0] -minnow[x,0])**2 + (self.agent_pos[1] -minnow[x,1])**2)
            
        #     dist_previous = np.sqrt((self.previous_pos[0] -minnow[x,0])**2 + (self.previous_pos[1] -minnow[x,1])**2)
        #     # if dist == 0:
        #     #     reward += 30
        #     # else:
        #     #     reward = 2000/dist
        #     if dist_previous < dist:
        #         reward -= 2

            # reward += 3*(dist_previous-dist)
        
            # if dist < dist_previous:
            #     self.dist_count += 1
            #     reward += 10
                
            # else:
            #     reward -= 15
            # # if not dist == 0:
            # #     reward += 10/(dist)

            if dist < 1.75:
                self.grid_state[minnow[x][0],minnow[x][1]] = 0
                reward += 100
                done = 1
            else:
                list_dist.append(dist)
        if list_dist:
            min_index = list_dist.index(min(list_dist))# check for shortest distance always returns lowest index
            rel_x_prev = minnow[min_index,0]- self.previous_pos[0]
            rel_y_prev = minnow[min_index,1] - self.previous_pos[1]

            rel_x = minnow[min_index,0]- self.agent_pos[0]
            rel_y = minnow[min_index,1] - self.agent_pos[1]


            if abs(rel_x) < abs(rel_x_prev):
                reward += 2
            elif rel_x == rel_x_prev:
                reward = reward
            else:
                reward -=2

            
            if abs(rel_y) < abs(rel_y_prev):
                reward += 2
            elif rel_y == rel_y_prev:
                reward = reward
            else:
                reward -=2

        else:
            rel_x = 0
            rel_y = 0
            reward += 50
        #out of bounds
        if self.agent_pos[0] == -1 or self.agent_pos[0] == 25:
            self.agent_pos = self.previous_pos
            # reward -= 10
            # done = 1
        if self.agent_pos[1] == -1 or self.agent_pos[1] == 50:
            self.agent_pos = self.previous_pos
            # reward -= 10
            # done = 1
        #wall checker if the action hits a wall revert to previous state(works)
        if self.grid_state[self.agent_pos[0],self.agent_pos[1]] == 1:
            self.agent_pos = self.previous_pos
            # reward -= 10
            # done = 1


        
            


        self.grid_state[self.previous_pos[0],self.previous_pos[1]] = 0
        self.grid_state[self.agent_pos[0],self.agent_pos[1]] = 3
        

        tmp2 = np.where(self.grid_state ==2) #finds minnow locations
        minnow = np.vstack((tmp2[0],tmp2[1])).T

        if not minnow.size > 0:
            done = 1

        info = {}


        return np.array([rel_x,rel_y]).flatten(), reward, done, info 

    def share_grid(self):

        return self.grid_state

    def receive_grid(self,grid_state):

        self.grid_state = grid_state

        return

    def render(self, mode='human'):
        path = '/home/aaron/spinningup/spinup/Results/Figures'
        name = 'fig_%s' %self.t

        #DO NOT CALL IN TRAINING
        #you will absolutely freeze your computer
        cmap = colors.ListedColormap(['blue', 'yellow','#c2c2c2','red','green','white','#ff9494','black'])
        bounds=[0,1,2,3,4,5,6,7,8]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        # fig = plt.figure()

        im = plt.matshow(np.flip(self.grid_state,axis = 0), interpolation='nearest', cmap = cmap,norm=norm)
        # title_txt = "Sharks: {}  Minnows: {}".format(self.eaten_minnows,0)
        # plt.title (title_txt)
        plt.savefig(osp.join(path,name))
        plt.show(block=False)

    def close(self):
        #not necessary if only running on cpu
        pass
        
        
