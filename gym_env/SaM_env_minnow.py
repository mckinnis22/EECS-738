import numpy as np
import gym
from gym import error, spaces
from gym import utils

import matplotlib.pyplot as plt
from matplotlib import colors

class SaMEnv_min(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SaMEnv_min,self).__init__()

        self.grid_state = np.zeros((25,50),dtype=float)
        # self.checked_space = np.zeros((25,50),dtype=float)
        self.action_space = spaces.Discrete(4) #defining the 4 possible movements per minnow

        #defines the grid we are playing in and the possible values for each
        self.observation_space = spaces.Discrete(4)
        self.dist_count = 0
        self.t = 1000
        self.number_minnow = 1
        self.number_shark = 1
        #self.observation_space = spaces.Discrete(1250)
        self.minnow1 = np.empty(2*self.number_minnow,)
        self.minnow0 = np.empty(2*self.number_minnow,)
        self.shark1 = np.empty(2*self.number_shark,)
        self.shark0 = np.empty(2*self.number_shark,)
        self.success_minnows = 0

    def reset(self):
        #this function creates a random game space for the sharks and minnows game
        #2 vertical walls of random length and location
        #6 static minnows
        #1 hungry shark
        #static end goal (not used)
        z = np.zeros((25,50))

        # -------- walls --------------------- #
        #creates wall starters
        walls = 0
        wall_seperation_length = np.int(np.floor(50/(walls+1)))
        wall_fudge_length = np.int(np.floor(wall_seperation_length/3))
        wall_seeds = np.arange(0,49,wall_seperation_length)
        wall_seeds = wall_seeds[1:walls+1] + np.random.randint(-wall_fudge_length,wall_fudge_length,size=1)
        z[wall_seeds,:] = 1

        wall_gap_locations = np.random.randint(3,21,size=walls)
        wall_gap_size = np.random.randint(3,5,size=walls)
        for i in range(walls):
            wall_gap = np.arange(np.clip(wall_gap_locations[i] - wall_gap_size[i],0,24), np.clip(wall_gap_locations[i] + wall_gap_size[i],0,24))
            z[wall_seeds[i],wall_gap] = 0

        # ------------ minnows --------------- 
        # minnow_y = np.random.random_integers(low = 2, high = 21,size = 1)
        # minnow_x = np.random.random_integers(low = 1, high = 20,size = 1)

        #need a logic to not start a minnow on a wall

        tmp2 = np.where(self.grid_state ==2) #finds minnow locations
        # minnow = np.vstack((tmp2[0],tmp2[1])).T

        # x = np.nonzero(z[minnow_y,minnow_x])      

        # z[minnow_y,minnow_x] = 2

        # ------------ sharks --------------- 
        # shark_y = np.random.random_integers(low = 4, high = 20,size = 2)
        # shark_x = np.random.random_integers(low = 15, high = 35,size = 2)

        # x = np.nonzero(z[shark_y,shark_x])

        # z[shark_y,shark_x] = 3

        tmp3 = np.where(self.grid_state ==3)
        
        self.minnow1 = np.vstack([tmp2[0],tmp2[1]])
        self.minnow0 = self.minnow1.T
        self.shark1 = np.array([tmp3[0],tmp3[1]]).T
        self.shark0 = self.shark1
        # Goal point
        goal_y = np.random.random_integers(low = 0, high = 23,size = 1)
        z[goal_y,49] = 4

        self.grid_state = z

        self.goal = np.argwhere(self.grid_state == 4)
        self.walls = np.argwhere(self.grid_state == 1)

        # r_goal1 = self.goal - self.minnow1
        # r_sharks1 = self.shark1.T - self.minnow1
        # minr_sharks1 = np.argmin(np.linalg.norm(r_sharks1,axis=1))
        # r_sharks1 = r_sharks1[minr_sharks1]

        r_1gx = np.zeros(self.number_minnow)
        r_1gy = np.zeros(self.number_minnow)
        r_1sx = np.zeros(self.number_minnow)
        r_1sy = np.zeros(self.number_minnow)

        # if self.walls.size == 0: #no walls
        #     min_wall1 = np.array([-1,-1])
        # else:
        #     min_wall1 = self.walls[np.argmin(np.linalg.norm(self.walls - self.minnow1,axis = 1))]
        # if np.argwhere(self.grid_state == 4).size == 0:
        #     print('ohno, safe zone is overwritten')

        return np.array([r_1gx,r_1gy,r_1sx,r_1sy]).T.flatten()

    def step(self,action):
        #this is the function where you need to propogate the agent as well as assign the reward for actions
        #Currently the reward is punishing movement around the board and slightly rewarding movement towards the
        #lower bound, once the lower bound is reach a large reward is given and the done flag is set to 1
        
        tmp2 = np.where(self.grid_state ==2)
        self.minnow1 = np.array([tmp2[0],tmp2[1]]).T
        self.number_minnow = len(self.minnow1)
        

        reward = -1 * np.ones(self.number_minnow,)
        done = 0
        dead = 0
        self.t -= 1
        
        self.minnow0 = self.minnow1
        self.shark0 = self.shark1

        tmp3 = np.where(self.grid_state ==3)
        self.shark1 = np.array([tmp3[0],tmp3[1]])

        shark_actions = np.array([[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]])
        minnow_actions = np.array([[1,0],[0,1],[-1,0],[0,-1],[0,0]]) 

        
        #Keep minnows on playing field
        
        #need to delete the actions of those who have been eaten

        


        self.minnow1 = self.minnow0 + minnow_actions[action]
        self.minnow1[:,1] = np.clip(self.minnow1[:,1], 0, 49)
        self.minnow1[:,0] = np.clip(self.minnow1[:,0], 0, 24)
        if len(self.minnow1) > 0:
            unq, count = np.unique(self.minnow1,axis = 0, return_counts=True)

            a  = unq[count >1]
            if len(a) > 0:
                for x in np.arange(0,np.shape(a)[0]):


                    idx = np.where(np.all(self.minnow1 == a[x,:], axis = 1))

                    self.minnow1[idx[0][1],:] = self.minnow0[idx[0][1],:] 

        #Punish even more if it gets eaten
        # if self.minnow1[0] == self.shark1[0] and self.minnow1[1] == self.shark1[1]:
        #     reward -= 50
        #     print('minnow eaten')      
        #     done = 1
        #     dead = 1
        r_g1x = np.zeros(self.number_minnow)
        r_g1y = np.zeros(self.number_minnow)
        r_s1x = np.zeros(self.number_minnow)
        r_s1y = np.zeros(self.number_minnow)

        got_to_goal = []
        if np.argwhere(self.grid_state == 4).size == 0:
            print('ohno, safe zone is overwritten')
        for y in np.arange(0,self.number_minnow):

            r_goal0 = self.goal - self.minnow0[y,:]
            r_goal1 = self.goal - self.minnow1[y,:]
            r_g1x[y] = r_goal1[0][0]
            r_g1y[y] = r_goal1[0][1]

            r_sharks0 = self.shark0.T - self.minnow0[y,:]
            r_sharks1 = self.shark1.T - self.minnow1[y,:]

            minr_sharks0 = np.argmin(np.linalg.norm(r_sharks0,axis=1))
            r_sharks0 = r_sharks0[minr_sharks0,:]
            minr_sharks1 = np.argmin(np.linalg.norm(r_sharks1,axis=1))
            r_s1x[y] = r_sharks1[minr_sharks1,0]
            r_s1y[y] = r_sharks1[minr_sharks1,1]

            #punish getting eaten by sharks
            if np.linalg.norm(r_sharks1) == 0:
                reward[y] -= 50
                
            elif abs(r_sharks1[0][0]) < abs(r_sharks0[0]):
                reward[y] -= 1
            elif abs(r_sharks1[0][1]) < abs(r_sharks0[1]):
                reward[y] -= 1
            else:
                reward[y] = reward[y]

            #provide reward for reaching goal
            if np.linalg.norm(r_goal1) == 0 and done == 0:
                reward[y] += 100
                self.success_minnows += 1
                self.grid_state[self.goal[0][0],self.goal[0][1]] = 4
                got_to_goal= y
                print('goal found')
            elif abs(r_goal1[0][0]) < abs(r_goal0[0][0]) and done == 0:
                reward[y] += 2
            elif abs(r_goal1[0][1]) < abs(r_goal0[0][1]) and done == 0:
                reward[y] += 2
            else:
                reward[y] -= 2

        self.grid_state[self.goal[0][0],self.goal[0][1]] = 4

        # self.grid_state[self.shark0[:,0],self.shark0[:,1]] = 6
        # self.grid_state[self.shark1[:,0],self.shark1[:,1]] = 3

        self.grid_state[self.minnow0[:,0],self.minnow0[:,1]] = 0

        self.minnow1 = np.delete(self.minnow1,got_to_goal,0)
        self.grid_state[self.minnow1[:,0],self.minnow1[:,1]] = 2

        info = {}
        return np.array([r_g1x,r_g1y,r_s1x,r_s1y]).T.flatten(), reward, done, info 

    def render(self, mode='human'):
        #call this function to render the board

        #DO NOT CALL IN TRAINING
        #you will absolutely freeze your computer
        cmap = colors.ListedColormap(['blue', 'yellow','#c2c2c2','red','green','white','#ff9494'])
        bounds=[0,1,2,3,4,5,6,7]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        fig = plt.figure(1)

        im = plt.matshow(self.grid_state, interpolation='nearest', cmap = cmap,norm=norm)
        plt.show(block=False)


    def share_grid(self):

        return self.grid_state

    def receive_grid(self,grid_state):

        self.grid_state = grid_state

        return

    def update_obs(self):
        
        tmp2 = np.where(self.grid_state ==2)
        self.minnow1 = np.array([tmp2[0],tmp2[1]]).T
        self.number_minnow = len(self.minnow1)

        r_g1x = np.zeros(self.number_minnow)
        r_g1y = np.zeros(self.number_minnow)
        r_s1x = np.zeros(self.number_minnow)
        r_s1y = np.zeros(self.number_minnow)


        if np.argwhere(self.grid_state == 4).size == 0:
            print('ohno, safe zone is overwritten')
        for y in np.arange(0,self.number_minnow):

            r_goal0 = self.goal - self.minnow0[y,:]
            r_goal1 = self.goal - self.minnow1[y,:]
            r_g1x[y] = r_goal1[0][0]
            r_g1y[y] = r_goal1[0][1]

            r_sharks0 = self.shark0.T - self.minnow0[y,:]
            r_sharks1 = self.shark1.T - self.minnow1[y,:]

            minr_sharks0 = np.argmin(np.linalg.norm(r_sharks0,axis=1))
            r_sharks0 = r_sharks0[minr_sharks0,:]
            minr_sharks1 = np.argmin(np.linalg.norm(r_sharks1,axis=1))
            r_s1x[y] = r_sharks1[minr_sharks1,0]
            r_s1y[y] = r_sharks1[minr_sharks1,1]

        return np.array([r_g1x,r_g1y,r_s1x,r_s1y]).T.flatten()


    def share_win(self):
        return self.success_minnows

    def close(self):
        #not necessary if only running on cpu
        pass
        
        
