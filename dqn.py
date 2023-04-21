from collections import deque
import random
import gym
import arm_2D_v2
import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import models, layers, optimizers ,regularizers


class DQN(object):
    def __init__(self):
        self.step = 0
        self.update_freq = 200                                        # model update frequency
        self.replay_size = 6000                                       # replay memory buffer size, 2000, 4000, 6000
        self.replay_queue = deque(maxlen=self.replay_size)            # store replay memory
        self.model = self.create_model()
        self.target_model = self.create_model()


    def create_model(self):
        # create neural network 
        STATE_DIM, ACTION_DIM = 7, 4 
        model = models.Sequential([
            layers.Dense(256, input_dim=STATE_DIM, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(ACTION_DIM, activation="linear")
        ])

        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.Adam(0.001))
        
        return model


    # epsilon-greedy behaviour with decay (step * decay constant)
    def act(self, s, epsilon=0.5):
        decay_constant = 0.0002

        # at the beginning, generate more states        
        if self.step == 0:
            if np.random.uniform() < 0.8:
                return np.random.choice([0, 1, 2, 3]) 
            return np.argmax(self.model.predict(np.array([s]))[0])
        
        # when epsilon is very small (less than 0.2)
        elif (epsilon - self.step * decay_constant) <= 0.2:
            if np.random.uniform() < 0.2:
                return np.random.choice([0, 1, 2, 3]) 
            return np.argmax(self.model.predict(np.array([s]))[0])
        
        # action is chosen based on decaying epsilon-greedy behaviour
        else:
            if np.random.uniform() < epsilon - self.step * decay_constant:
                return np.random.choice([0, 1, 2, 3]) 
            return np.argmax(self.model.predict(np.array([s]))[0])


    # save the trained model
    def save_model(self, file_path='models/final_arm_model_dqn_FCNN2.h5'):
        print('Model saved')
        self.model.save(file_path)


    # store experiences in a replay memory, randomly sampled during nn training
    def remember(self, s, a, next_s, reward):
        """Additional rewards when robotic arm approaches target/goal for faster convergence"""
        # observation space : [joint_1_theta, joint_2_theta, target_tip_dis_x, target_tip_dis_y, goal_tip_dis_x, goal_tip_dis_y, PnP_action]
        
        # if arm approaches target and PnP_action == 0
        if abs(next_s[2]) <= 40 and abs(next_s[3]) <= 40 and next_s[6] == 0:
            reward += 0.5
        # if arm approaches goal and PnP_action == 1
        elif abs(next_s[4]) <= 100 and abs(next_s[5]) <= 100 and next_s[6] == 1:
            reward += 0.5

        self.replay_queue.append((s, a, next_s, reward))              # store replay memory


    # batch_size, lr: learning rate, dis_factor: discount factor
    def train(self, batch_size=64, lr=0.9, dis_factor=0.9):
        if len(self.replay_queue) < self.replay_size:
            return
        self.step += 1
        # update_freq，assign the model weight to target_model
        if self.step % self.update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        replay_batch = random.sample(self.replay_queue, batch_size)
        s_batch = np.array([replay[0] for replay in replay_batch])
        next_s_batch = np.array([replay[2] for replay in replay_batch])

        Q = self.model.predict(s_batch)                           # Q is calculated using main model
        Q_next = self.target_model.predict(next_s_batch)          # Q_next is calculated using target model (prediction of next step)

        # Q-learning
        for i, replay in enumerate(replay_batch):
            _, a, _, reward = replay
            Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward + dis_factor * np.amax(Q_next[i]))      # update Q table using update rule
 
        # pass into the network for training
        self.model.fit(s_batch, Q, verbose=0)


env = arm_2D_v2.Arm_2D_v2()

episodes = 400      # training time
score_list = []     # record all score
termination_score = 50
current_mean = 0
agent = DQN()
for i in range(episodes):
    step_count = 0
    s = env.reset()
    score = 0
    while True:
        # if i % 10 == 0:
        #     env.render()
        # else:
        #     env.close() 
        a = agent.act(s)                           # choose an action based on epsilon-greedy behaviour
        next_s, reward, done, _ = env.step(a)      # take action to get the next step and its reward, info
        agent.remember(s, a, next_s, reward)       # store experience in replay memory
        agent.train()                              # train : model weight adjusting, Q learning
        score += reward                            # acumulate score
        s = next_s                                 # update state
        if done or step_count == 1000:             # episode termination
            score_list.append(score)               # store score in score_list
            print('episode:', i, 'score:', score, 'max:', max(score_list), 'mean',current_mean)
            break
        step_count += 1

    # stop and save the model when the average score of the last 50 times is greater than termination score
    current_mean = np.mean(score_list[-50:])       # calculate the mean of the last 50 scores
    if current_mean >= termination_score:          # training termination condition
        print("Training is completed")             
        agent.save_model()                         # save the model 
        break

agent.save_model()
env.close()

arr_rounded = np.round(score_list, decimals=2)
print(arr_rounded)

with open("figs/FCNN2.txt", "w") as file:
    file.write(str(arr_rounded))

plt.plot(score_list, color='green')
plt.xlabel('Episodes')
plt.ylabel('Scores')
plt.savefig('figs/FCNN2.jpg')

