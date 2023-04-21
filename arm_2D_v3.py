import gym
import math 
import random
import pygame
import numpy as np
import time
from gym import utils
from gym import error, spaces
from gym.utils import seeding
from scipy.spatial.distance import euclidean

class Arm_2D_v3(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):
            self.set_window_size([600,600])                 # environment size (pixel)
            self.set_link_properties([100,100])             # robot arm length (pixel)
            self.set_increment_rate(0.1)                    # +/- joint angle (0.1 rad) 
            self.num_target = 5                             # number of targets
            self.all_target_pos = self.generate_multiple_targets(self.num_target)       # generate all target positions
            self.target_pos, self.remaining_targets = self.choose_target(self.all_target_pos)   # choose the target nearest to the goal, update the all_target_pos with leftover targets
            self.PnP_action = 0                             # pick and place flag

            self.action = {0: "INC_J1",                     # action space
                           1: "DEC_J1",
                           2: "INC_J2",
                           3: "DEC_J2"}
            
            self.theta_last = [-1000, -1000]                # record the last joint angles

            # state number:7, action number: 4
            # observation: [joint_1_theta, joint_2_theta, target_tip_dis_x, target_tip_dis_y, goal_tip_dis_x, goal_tip_dis_y, PnP_action]
            # self.observation_space = spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, shape=(7,), dtype=np.float32)
            self.observation_space = spaces.Box(np.float32(np.finfo(np.float32).min), np.float32(np.finfo(np.float32).max), shape=(7,))
            self.action_space = spaces.Discrete(len(self.action))

            self.current_error = -math.inf                  # record distance error 
            self.seed()
            self.viewer = None


    # set window size
    def set_window_size(self, window_size):
        self.window_size = window_size
        self.centre_window = [window_size[0]//2, window_size[1]//2]


    # set joints increment rate
    def set_increment_rate(self, rate):
        self.rate = rate


    # define joints properties 
    def set_link_properties(self, links):
        self.links = links
        self.n_links = len(self.links)          # number of joints 
        self.min_theta_1 = math.radians(0)      # limit the range of joint movement 
        self.max_theta_1 = math.radians(180)
        self.min_theta_2 = math.radians(-180)  
        self.max_theta_2 = math.radians(180)
        # self.theta = self.generate_random_angle()       # initialize random arm pose
        self.theta = self.arm_initial_angle()           # initialize fixed arm pose 
        self.max_length = sum(self.links)


    # random angle generator 
    def generate_random_angle(self):
        theta = np.zeros(self.n_links)
        theta[0] = random.uniform(self.min_theta_1, self.max_theta_1)   # random angle from (0-180) <-> (0-pi)
        theta[1] = random.uniform(self.min_theta_2, self.max_theta_2)   # random angle from (-180-180) <-> (-pi-pi)
        return theta
    

    # initialize fixed arm pose
    def arm_initial_angle(self):
        theta = np.zeros(self.n_links)
        theta[0] = 1.57   # initial pose
        theta[1] = 0
        return theta


    # random position of the target 
    def generate_random_pos(self):
        theta = self.generate_random_angle()        # random angle that the arm can reach
        P = self.forward_kinematics(theta)          # calculate the position from angle
        pos = np.array([P[-1][0,3], P[-1][1,3]])    # get the x, y in the matrix
        while not (-100 < pos[0] < 100 and 100 < pos[1] < 170): # while pos is not within the assembly line area
            new_seed = self.seed()[0]                           # generate a new seed
            self.seed(new_seed)                                 # set the new seed for the env
            theta = self.generate_random_angle()                # random angle that the arm can reach
            P = self.forward_kinematics(theta)                  # calculate the position from angle
            pos = np.array([P[-1][0,3], P[-1][1,3]])            # get the x,y in the matrix
        return pos


    # generate multiple targets 
    """ def generate_multiple_targets(self, num_target):
        targets = []
        for i in range(num_target):
            targets.append(self.generate_random_pos())
            new_seed = self.seed()[0]
            self.seed(new_seed)
        return targets """
    

    # generate multiple targets without collision 
    def generate_multiple_targets(self, num_target):
        targets = []
        for i in range(num_target):
            while True:
                new_target = self.generate_random_pos() # generate random target position
                collision = False
                for target in targets:
                    # calculate the distance between new_target and existing targets
                    distance = euclidean(new_target, target)
                    if distance < 40:  # check for collision
                        collision = True
                        break
                if not collision:  # if no collision, add new_target to targets
                    targets.append(new_target)
                    break
            new_seed = self.seed()[0]   # generate a new seed
            self.seed(new_seed)         # set the new seed for the env
        return targets


    # choose nearest target
    def choose_target(self, all_target_pos):           # return the target that is closest to the goal, and leftover targets 
        if all_target_pos:
            distances_from_goal = []
            for target in all_target_pos:
                distances_from_goal.append(euclidean([200, 0], target))
            sorted_targets = [pos for _, pos in sorted(zip(distances_from_goal, all_target_pos))]
            chosen_target = sorted_targets[0]
            sorted_targets.pop(0)
            return chosen_target, sorted_targets


    # updating targets after pick 
    def Update_target_after_pick(self):
        self.all_target_pos = self.remaining_targets


    # updating targets after place 
    def Update_target_after_place(self):
        self.target_pos, self.remaining_targets = self.choose_target(self.all_target_pos)
        

    # forward kinematics calculation 
    def forward_kinematics(self, theta):
        P = []
        P.append(np.eye(4))
        for i in range(0, self.n_links):
            R = self.rotate_z(theta[i])
            T = self.translate(self.links[i], 0, 0)
            P.append(P[-1].dot(R).dot(T))
        return P


    # rotation matrix
    def rotate_z(self, theta):
        rz = np.array([[np.cos(theta), - np.sin(theta), 0, 0],
                        [np.sin(theta), np.cos(theta), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        return rz


    # translation matrix
    def translate(self, dx, dy, dz):
        t = np.array([[1, 0, 0, dx],
                        [0, 1, 0, dy],
                        [0, 0, 1, dz],
                        [0, 0, 0, 1]])
        return t


    # random seed 
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    

    # inverse the angle 
    def inverse_theta(self, theta):
        new_theta = theta.copy()
        for i in range(theta.shape[0]):
            new_theta[i] = -1*theta[i]
        return new_theta


    # Pygame drawing section: 
    # draw the arm 
    def draw_arm(self, theta):
        LINK_COLOR = (238, 201, 0)
        JOINT_COLOR = (105, 105, 105)
        TIP_COLOR = (34, 139, 34)
        theta = self.inverse_theta(theta)
        P = self.forward_kinematics(theta)
        origin = np.eye(4)
        origin_to_base = self.translate(self.centre_window[0],self.centre_window[1],0)
        base = origin.dot(origin_to_base)
        F_prev = base.copy()
        for i in range(1, len(P)):
            F_next = base.dot(P[i])
            pygame.draw.line(self.screen, LINK_COLOR, (int(F_prev[0,3]), int(F_prev[1,3])), (int(F_next[0,3]), int(F_next[1,3])), 5)
            pygame.draw.circle(self.screen, JOINT_COLOR, (int(F_prev[0,3]), int(F_prev[1,3])), 10)
            F_prev = F_next.copy()
        pygame.draw.circle(self.screen, TIP_COLOR, (int(F_next[0,3]), int(F_next[1,3])), 8)


    # draw the target on the assembly line 
    def draw_target(self, single_target):          # added argument single_target for easy use
        TARGET_COLOR = (99, 184, 255)
        origin = np.eye(4)
        origin_to_base = self.translate(self.centre_window[0], self.centre_window[1], 0)
        base = origin.dot(origin_to_base)
        base_to_target = self.translate(single_target[0], -single_target[1], 0)
        target = base.dot(base_to_target)
        pygame.draw.circle(self.screen, TARGET_COLOR, (int(target[0,3]),int(target[1,3])), 20)


    # draw multiple targets
    def draw_multiple_targets(self,targets):
        for i in targets:
            self.draw_target(i)
    

    # draw the target on the arm 
    def draw_target_on_tip(self):
        TARGET_COLOR = (99, 184, 255)
        pygame.draw.circle(self.screen, TARGET_COLOR, (int(self.tip_pos[0])+300, 300-int(self.tip_pos[1])), 20)


    # draw the assembly line 
    def draw_assembly_line(self):
        al_color = (205, 201, 201)
        al_width = 340
        al_height = 140
        al_x = 130
        al_y = 80
        pygame.draw.rect(self.screen, al_color, (al_x, al_y, al_width, al_height))

    
    # draw the goal 
    def draw_goal(self):
        goal_color = (205, 133, 63)
        goal_width = 100
        goal_height = 100
        goal_x = 450
        goal_y = 250
        pygame.draw.rect(self.screen, goal_color, (goal_x, goal_y, goal_width, goal_height))


    @staticmethod
    # computes the angle in radians, to determin the quadrant of the angle 
    def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))


    # step function 
    def step(self, action):
        # define actions 
        if self.action[action] == "INC_J1":
            self.theta[0] += self.rate
        elif self.action[action] == "DEC_J1":
            self.theta[0] -= self.rate
        elif self.action[action] == "INC_J2":
            self.theta[1] += self.rate 
        elif self.action[action] == "DEC_J2":
            self.theta[1] -= self.rate

        # limit and normalize the angle 
        self.theta[0] = np.clip(self.theta[0], self.min_theta_1, self.max_theta_1)
        self.theta[1] = np.clip(self.theta[1], self.min_theta_2, self.max_theta_2)
        self.theta[0] = self.normalize_angle(self.theta[0])
        self.theta[1] = self.normalize_angle(self.theta[1])

        # calculate the tip position 
        P = self.forward_kinematics(self.theta)
        self.tip_pos = [P[-1][0,3], P[-1][1,3]]
        
        # calculate the xy distance between the target and the tip 
        target_tip_dis_x = self.target_pos[0] - self.tip_pos[0]
        target_tip_dis_y = self.target_pos[1] - self.tip_pos[1]

        # calculate the xy distance between the goal and the tip 
        goal_tip_dis_x = 200 - self.tip_pos[0]
        goal_tip_dis_y = 0 - self.tip_pos[1]

        # calculate the distance between the target/goal and the tip
        dis_err_target = euclidean(self.target_pos, self.tip_pos)
        dis_err_goal = euclidean([200, 0], self.tip_pos)

        # the arm is looking for the target, PnP_action is 0 now
        if self.PnP_action == 0:
            # not return distance between the goal and the tip if not picking
            goal_tip_dis_x = 0 
            goal_tip_dis_y = 0

            reward = 0      # 0 reward for other situations 

            if dis_err_target >= self.current_error:    # if far from the target, give a penalty
                reward = -0.2

            # if reach the target, give large reward, and set PnP_action to 1
            close_enough_tar = 20
            if (dis_err_target > -close_enough_tar and dis_err_target < close_enough_tar): 
                print("Pick target!!!") 
                reward = 10                
                self.PnP_action = 1                 # self PnP_action to 1, continue find goal 
                self.Update_target_after_pick()     # update targets

            # if staying in place, give a large penalty 
            if (abs(self.theta[0] - self.theta_last[0]) < 0.001 and abs(self.theta[1] - self.theta_last[1]) < 0.001): 
                reward = -1

            self.current_error = dis_err_target     # update current distance error between the target and the tip

        # the arm is looking for the goal, PnP_action is 1 now
        elif self.PnP_action == 1:
            # not return distance between the target and the tip if picking
            target_tip_dis_x = 0 
            target_tip_dis_y = 0

            reward = 0     # 0 reward for other situations

            if dis_err_goal >= self.current_error:      # if far from the target, give a penalty
                reward = -0.2

            # if reach the goal, give large reward, and set PnP_action to 0
            close_enough_goal = 50 
            if (dis_err_goal > -close_enough_goal and dis_err_goal < close_enough_goal): 
                print("Place target!!!")
                reward = 10
                self.PnP_action = 0         # self PnP_action to 0, continue find target
                
                if len(self.remaining_targets)==0:      # only reset the targets if there are no targets left to PnP
                    self.all_target_pos = self.generate_multiple_targets(self.num_target)   # new set of target positions are generated
                    self.target_pos, self.remaining_targets = self.choose_target(self.all_target_pos)   # choose nearest target 
                else:
                    self.Update_target_after_place()    # update targets

            # if staying in place, give a large penalty 
            if (abs(self.theta[0] - self.theta_last[0]) < 0.001 and abs(self.theta[1] - self.theta_last[1]) < 0.001):
                reward = -1

            self.current_error = dis_err_goal           # update current distance error between the goal and the tip

        # record the current theta
        self.theta_last[0] = self.theta[0]
        self.theta_last[1] = self.theta[1]
        
        self.current_score += reward    # accumulative reward
        # print("Current score : ", self.current_score)

        # finish an epoch if accumulative reward is too small, or finished PnP several times
        if self.current_score <= -20 or self.current_score >= 100: 
            done = True
        else:
            done = False

        # observation space 
        observation = np.hstack((self.theta, target_tip_dis_x, target_tip_dis_y, goal_tip_dis_x, goal_tip_dis_y, self.PnP_action))

        info = {
            'theta': self.theta,
            'target_tip_dis_x': target_tip_dis_x,
            'target_tip_dis_y': target_tip_dis_y,
            'goal_tip_dis_x': goal_tip_dis_x,
            'goal_tip_dis_y': goal_tip_dis_y,
            'PnP_action': self.PnP_action
        }
        return observation, reward, done, info


    # reset function 
    def reset(self):        
        self.all_target_pos = self.generate_multiple_targets(self.num_target)       # new set of target positions are generated
        self.target_pos, self.remaining_targets = self.choose_target(self.all_target_pos)   # choose nearest target s

        self.theta = self.arm_initial_angle()           # reset arm position 
        self.theta_last = [-1000, -1000]                # reset theta buffer 
        self.current_score = 0                          # reset score buffer 
        P = self.forward_kinematics(self.theta)         # calculate current tip position 
        self.tip_pos = [P[-1][0,3], P[-1][1,3]]

        target_tip_dis_x = self.target_pos[0] - self.tip_pos[0]     # get xy distance between the target and the tip 
        target_tip_dis_y = self.target_pos[1] - self.tip_pos[1]

        goal_tip_dis_x = 200 - self.tip_pos[0]      # get xy distance between the goal and the tip 
        goal_tip_dis_y = 0 - self.tip_pos[1]

        self.PnP_action = 0     # reset PnP_action flag 

        observation = np.hstack((self.theta, target_tip_dis_x, target_tip_dis_y, goal_tip_dis_x, goal_tip_dis_y, self.PnP_action))
        return observation


    # rendering 
    def render(self, mode='human'):
        SCREEN_COLOR = (193, 255, 193)
        if self.viewer == None:
            pygame.init()
            pygame.display.set_caption("RobotArm-Env")
            self.screen = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()
            self.viewer = 1
        self.screen.fill(SCREEN_COLOR)
        self.draw_assembly_line()
        self.draw_goal()

        if self.PnP_action == 0:
            self.draw_multiple_targets(self.all_target_pos)
        elif self.PnP_action == 1:
            self.draw_multiple_targets(self.remaining_targets)
            self.draw_target_on_tip()

        self.draw_arm(self.theta)
        self.clock.tick(60)
        pygame.display.flip()


    # close environment 
    def close(self):
        if self.viewer != None:
            self.viewer = None
            pygame.quit()

