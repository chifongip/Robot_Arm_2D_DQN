import time
import gym
import arm_2D_v2
import numpy as np
from tensorflow.keras import models


env = arm_2D_v2.Arm_2D_v2()         # initial environment 
model = models.load_model('models/final_arm_model_dqn_LSTM.h5')   # load model 

success = 0 

for i in range(100):
    s = env.reset()
    score = 0
    while True:
        # env.render()
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

