import time
import gym
import arm_2D_v3
import numpy as np
import imageio
from tensorflow.keras import models


env = arm_2D_v3.Arm_2D_v3()         # initial environment 
model = models.load_model('models/final_arm_model_dqn_FCNN2.h5')   # load model 

success = 0
images = []  # create an empty list to store images for gif
for i in range(1):
    s = env.reset()
    score = 0
    while True:
        img = env.render()
        images.append(img)  # append image to the list
        a = np.argmax(model.predict(np.array([s]))[0])
        s, reward, done, _ = env.step(a)
        score += reward
        if done:
            if score > 0:
                success += 1
            print('score:', score)
            break

# save images as gif
imageio.mimsave('rendering.gif', images, fps=30)

print('Success rate: ', success)

env.close()

