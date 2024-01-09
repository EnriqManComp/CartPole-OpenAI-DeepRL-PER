import gym
import tensorflow as tf
from duelingNetwork import duelingDQN
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder


env = gym.make('CartPole-v1', render_mode='rgb_array')
vid = VideoRecorder(env, 'video/video.mp4')

model_path = "networks/d3dqn_model1"      
q_net = tf.keras.models.load_model(model_path)



for epis in range(0,1):
        
        observation = env.reset()
        observation = observation[0]   
        
        #env.render()
        
        
        done = False
        score = 0.0
        
        while not done:
            vid.capture_frame()
            state = np.array(observation)
            state = np.expand_dims(state, axis = 0)            
            actions = q_net.predict(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]

            # Perform the action in the enviroment
            next_observation, reward, done, _, _ = env.step(action)

            # Cumulative reward
            score += reward
            
            

            if score == 400.0:
                break
            # Updating next step observations
            observation = next_observation
print(score)  
vid.close()
env.close()