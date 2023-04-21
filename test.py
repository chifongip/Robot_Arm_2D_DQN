import gym
import arm_2D_v3
import time
import pygame

env = arm_2D_v3.Arm_2D_v3()

observation = env.reset()

while True:
    env.render()
    action = env.action_space.sample()
    
    observation, reward, done, info = env.step(action)
    print(observation)
    
    if done:
        print('done')
        break

env.close()
