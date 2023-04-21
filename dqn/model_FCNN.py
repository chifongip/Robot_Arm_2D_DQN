from collections import deque
import random
import gym
import arm_2D_v2
import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, optimizers


class DQN(object):
    def __init__(self):
        self.step = 0
        self.update_freq = 200      # model update frequency
        self.replay_size = 6000     # training set size, 2000, 4000
        self.replay_queue = deque(maxlen=self.replay_size)
        self.model = self.create_model()
        self.target_model = self.create_model()


    def create_model(self):
        """Create a neural network with 100 hidden neurons"""
        # create neural network 
        STATE_DIM, ACTION_DIM = 7, 4 

        model = models.Sequential([
            layers.Dense(100, input_dim=STATE_DIM, activation='relu'),            
            layers.Dense(100, activation='relu'),
            layers.Dense(100, activation='relu'),
            layers.Dense(ACTION_DIM, activation="linear")
        ])

        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.Adam(0.001))
        
        # model.summary()

        return model


    # epsilon-greedy?????
    def act(self, s, epsilon=0.5): 
        # At the beginning, generate more states        
        if self.step == 0:
            if np.random.uniform() < 0.8:
                return np.random.choice([0, 1, 2, 3]) 
            return np.argmax(self.model.predict(np.array([s]))[0])
        elif (epsilon - self.step * 0.0002) <= 0.2: 
            if np.random.uniform() < 0.2: 
                return np.random.choice([0, 1, 2, 3]) 
            return np.argmax(self.model.predict(np.array([s]))[0])
        else:
            if np.random.uniform() < epsilon - self.step * 0.0002:
                return np.random.choice([0, 1, 2, 3]) 
            return np.argmax(self.model.predict(np.array([s]))[0])
    

    def save_model(self, file_path='final_arm_model_dqn_FCNN.h5'):
        print('model saved')
        self.model.save(file_path)


    # replay memory? not sure how to use now
    def remember(self, s, a, next_s, reward):
        """History, distance <= (threshold), gives additional rewards, fast convergence"""
        # [joint_1_theta, joint_2_theta, target_tip_dis_x, target_tip_dis_y, goal_tip_dis_x, goal_tip_dis_y, PnP_action]
        if abs(next_s[2]) <= 40 and abs(next_s[3]) <= 40 and next_s[6] == 0:
            reward += 0.5
        elif abs(next_s[4]) <= 100 and abs(next_s[5]) <= 100 and next_s[6] == 1:
            reward += 0.5
        self.replay_queue.append((s, a, next_s, reward))


    # learning rate and gamma!!!!! batch_size = 64, lr = 1, factor = 0.95
    def train(self, batch_size=64, lr=0.9, factor=0.9):
        if len(self.replay_queue) < self.replay_size:
            return
        self.step += 1
        # update_freq，assign the model weight to target_model
        if self.step % self.update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        replay_batch = random.sample(self.replay_queue, batch_size)
        s_batch = np.array([replay[0] for replay in replay_batch])
        next_s_batch = np.array([replay[2] for replay in replay_batch])

        Q = self.model.predict(s_batch)
        Q_next = self.target_model.predict(next_s_batch)

        # Q-learning
        for i, replay in enumerate(replay_batch):
            _, a, _, reward = replay
            Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward + factor * np.amax(Q_next[i]))
 
        # pass into the network for training
        self.model.fit(s_batch, Q, verbose=0)


# env = gym.make('MountainCar-v0')
env = arm_2D_v2.Arm_2D_v2()

episodes = 400      # training time
score_list = []     # record all score
agent = DQN()
for i in range(episodes):
    k = 0
    s = env.reset()
    score = 0
    while True:
        """ if i % 10 == 0:
            env.render()
        else:
            env.close() """
        a = agent.act(s)
        next_s, reward, done, _ = env.step(a)
        agent.remember(s, a, next_s, reward)
        agent.train()
        score += reward
        s = next_s
        if done or k == 1000:
            score_list.append(score)
            print('episode:', i, 'score:', score, 'max:', max(score_list))
            break
        k += 1
    # Stop and save the model when the average score of the last 30 times is greater than 50
    if np.mean(score_list[-50:]) >= 50:
        agent.save_model()
        break

agent.save_model()

env.close()

arr_rounded = np.round(score_list, decimals=2)
print(arr_rounded)

with open("FCNN.txt", "w") as file:
    file.write(str(arr_rounded))

plt.plot(score_list, color='green')
plt.xlabel('Episodes')
plt.ylabel('Scores')
plt.savefig('FCNN.jpg')

