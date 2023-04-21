import time
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

print('Success rate: ', success)

env.close()

