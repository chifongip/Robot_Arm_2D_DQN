# Robot_Arm_2D_DQN
NUS ME5406 Final Project  
<img src="https://github.com/chifongip/Robot_Arm_2D_DQN/blob/main/5_objects.gif" width="400" height="400"/>

# Overview
This project uses DQN algorithms for reinforcement learning of Robotic Arm to perform Pick-and-Place (PnP) task. The environment used for training can be found in arm_2D_v2 which is used to train DQN models and the environment for testing using multiple targets can be found in arm_2D_v3.

# Main Files
- models      : all trained models
- arm_2D_v2   : single target, single goal env, used for training DQN
- dqn.py      : modify and run the file to train, result is stored in models folder
- test_dqn_v2.py : test the trained model
- arm_2D_v3   : multiple targets, single goal env, not for training
- test_dqn_v3.py : test the trained model to PnP multiple targets

# Packages needed
gym, pygame, tensorflow, numpy, scipy

# Testing 
Run test_dqn_v3.py to test multiple objects PnP.  
The number of objects can be changed (1-8 for this environment size)  
The used model can be changed (refer to the following list)  
## Trained model 
FCNN: final_arm_model_dqn_FCNN1.h5 / final_arm_model_dqn_FCNN2.h5  
CNN: final_arm_model_dqn_CNN1.h5 / final_arm_model_dqn_CNN2.h5  
LSTM: final_arm_model_dqn_LSTM.h5  
## Testing code 
```
import gym
import arm_2D_v3
import numpy as np
from tensorflow.keras import models


env = arm_2D_v3.Arm_2D_v3()         # initial environment 
env.num_target = 5                  # choose object number (1-8) 
model = models.load_model('models/final_arm_model_dqn_CNN2.h5')   # load model 

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
