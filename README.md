# Robot_Arm_2D_DQN
NUS ME5406 Final Project

# Overview
This project uses DQN algorithms for reinforcement learning of Robotic Arm to perform Pick-and-Place (PnP) task. The environment used for training can be found in arm-v2/arm_2D_v2 which is used to train DQN models and the environment for testing using multiple targets can be found in arm-v3/arm_2D_v3.

# Main Files
Models             : All trained models

arm-v2/arm_2D_v2   : single target, single goal env, used for training DQN
arm-v2/dqn.py      : modify and run the file to train, result is stored in Models folder
arm-v2/test_dqn.py : test the trained model

arm-v3/arm_2D_v3   : multiple targets, single goal env, not for training
arm-v3/test_dqn.py : test the trained model to PnP multiple targets

# Packages needed
gym, pygame, tensorflow, numpy, scipy

## testing code
```
import gym
import arm_2D_v3
import numpy as np
from tensorflow.keras import models


env = arm_2D_v3.Arm_2D_v3()         # initial environment 
model = models.load_model('models/final_arm_model_dqn_LSTM.h5')   # load model 

success = 0 

for i in range(10):
    s = env.reset()
    score = 0
    while True:
        env.render()
        a = np.argmax(model.predict(np.array([s]))[0])      # predict action through model 
        s, reward, done, _ = env.step(a)
        score += reward
        if done:
            if score > 0:
                success += 1
            print('score:', score)
            break

print('Success rate: ', success*10, '%')

env.close()
```
