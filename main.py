############ LIBRARIES ###############

import gym
from duelingNetwork import duelingDQN
import tensorflow as tf
from PER import PER
from uniform_exp_replay import ReplayBuffer
import numpy as np
import pandas as pd

########## CONSTANTS ##################

EPISODES = 10000
LEARNING_RATE = 0.00025
REPLAY_INIT_SIZE = 500
BATCH_SIZE = 64
DISCOUNTED_FACTOR = 0.99

start_epsilon = 1.0
final_epsilon = 0.1
epsilon_decay = 0.00004 

########### ALGORITHM CODE ################

# Train or Test 
train = True

if train:
    # Creating enviroment
    env = gym.make('CartPole-v1')


    # Creating dueling Network
    dueling_object = duelingDQN(4, 256, 256, 2, LEARNING_RATE)
    q_net = dueling_object.model
    # Creating target dueling network
    target_dueling_object = duelingDQN( 4, 256, 256, 2, LEARNING_RATE)
    q_target_net = target_dueling_object.model
    # Copy weights

    q_target_net.set_weights(q_net.get_weights())
    # Creating memory object
    # Uniform Experience Replay (UER) or First Approach using Experience Replay
    #memory = ReplayBuffer(1_000_000)
    # Prioritized Experience Replay (PER)
    memory = PER(1_000_000)
    # Epsilon
    epsilon = start_epsilon
    # Goal of the agent
    goal = 200

    # Network Saved number f
    f = 1
    score_record = []
    mse_record = []

    for epis in range(1, EPISODES+1):
        print("EPISODE: ", epis)    
        
        observation = env.reset()
        observation = observation[0]   
        done = False
        # Cumulative reward
        score = 0.0    

        mse_losses = []    
        
        while not done:

            if np.random.rand() <= epsilon:
                action = np.random.choice(2)           
            else:
                state = np.array(observation)
                state = np.expand_dims(state, axis = 0)            
                actions = q_net.predict(state)
                action = tf.math.argmax(actions, axis=1).numpy()[0] 
                

            # Perform the action in the enviroment
            next_observation, reward, done, _, _ = env.step(action)

            # Cumulative reward
            score += reward            

            # Experience 
            experience = np.array([observation, reward, action, next_observation, done], dtype=object)        
            experience = experience.reshape((1,5))        

            # Adding experience
            memory.add(experience)

            # Start to train the dueling Network
            # For UER
            #if len(memory.storage) >= REPLAY_INIT_SIZE:
            if memory.tree.data_ptr:
                # Sample the memory
                # For PER
                weights, tree_idx, minibatch = memory.sample(BATCH_SIZE) 
                # For UER
                #minibatch = memory.sample(BATCH_SIZE)                
                
                # Preprocessing experience data
                states = np.empty((BATCH_SIZE, minibatch[0,0].size))            
                next_states = np.empty((BATCH_SIZE, minibatch[0,3].size))
                for batch in range(BATCH_SIZE):
                    states[batch,:] = minibatch[batch,0]
                    next_states[batch,:] = minibatch[batch,3]
                # Predictions with q_net   
                q_preds = q_net(states)            
                # Getting Q's max
                q_preds = tf.math.reduce_max(q_preds, axis=1).numpy()  
                # Prediction the next actions
                next_actions = tf.math.argmax(q_net(next_states), axis=1).numpy() 
                # Estimation of the next Q
                q_next_preds = q_target_net(next_states)

                q_target = np.empty_like(next_actions)       
                errors = np.empty_like(next_actions) 

                for idx in range(BATCH_SIZE):
                    # Pure rewards
                    q_target_value = minibatch[idx,1]                     
                    # If not done (Q = R + gamma*(Q'(Q(s',a')))) else Q = R
                    if not minibatch[idx, -1]:  
                        # Evaluation of the next action predictions
                        q_action_evaluated = q_next_preds[idx][next_actions[idx]]
                        # Bellman ecuation variation
                        q_target_value += DISCOUNTED_FACTOR*q_action_evaluated                                
                    # Estimations of Qs
                    q_target[idx] = q_target_value
                    # Computing absolute error for new priorities
                    errors[idx] = np.abs(q_preds[idx] - q_target_value)
                # Training the dueling network             
                # Using PER
                losses = q_net.train_on_batch(x= states, y= q_target, sample_weight=weights, return_dict=True)
                # Using UER
                #losses = q_net.train_on_batch(x= states, y= q_target, return_dict=True)
                # Updating losses (for PER)
                memory.update(tree_idx, errors)                    
                mse_losses.append(losses['loss'])
                # Decrement epsilon value
                epsilon = epsilon - epsilon_decay if epsilon > final_epsilon else final_epsilon
                # Updating q_target_network weights using Polyak's average
                target_dueling_object.soft_update(q_net.get_weights())          

            # Updating next step observations
            observation = next_observation

        score_record.append(score)
        mse_record.append(np.mean(mse_losses))
        avg_score = np.mean(score_record[-50:])

        # Saving network
        if (score >= 200) or (avg_score >= 150.0):
            q_net.save(("networks/d3dqn_model{0}".format(f)))
            with open("records/save_network.txt", 'a') as file:
                file.write("Save {0} - Episode {1}/{2}, Score: {3}, Epsilon: {4}, AVG Score: {5}\n".format(f, epis, EPISODES, score, epsilon, avg_score))
            f += 1
            print("Network saved")
        if epis == EPISODES:
            q_net.save(("networks/d3dqn_model{0}".format(f)))
            with open("records/save_network.txt", 'a') as file:
                file.write("Save {0} - Episode {1}/{2}, Score: {3}, Epsilon: {4}, AVG Score: {5}\n".format(f, epis, EPISODES, score, epsilon, avg_score))
            f += 1
            print("Network saved")

        if (epis % 50) == 0:
            record = pd.DataFrame(score_record)
            record = record.T 
            record['avg_score'] = avg_score
            mse = pd.DataFrame(mse_record)
            mse = mse.T        
            record = pd.concat([record, mse], axis=1)
            record.to_csv("records/records.csv", mode='a', header=False, index=False)
            mse_record = []
            score_record = []
else:
    env = gym.make("CartPole-v1", render_mode='human')   
    
    # Loading Dueling Network
    model_path = "networks/d3dqn_model7"      
    q_net = tf.keras.models.load_model(model_path)   
    score_record = []    
    for epis in range(1,11):
        print("EPISODE: ", epis)    
        # Reset enviroment
        observation = env.reset()
        observation = observation[0]
        env.render()   
        
        done = False
        # Cumulative reward
        score = 0.0
        
        

        while not done:

            state = np.array(observation)
            state = np.expand_dims(state, axis = 0)            
            actions = q_net.predict(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]

            # Perform the action in the enviroment
            next_observation, reward, done, _, _ = env.step(action)

            # Cumulative reward
            score += reward
            

            if score == 200.0:
                break
            # Updating next step observations
            observation = next_observation
        score_record.append(score)        
    # Saving the test record   
    record = pd.DataFrame(score_record)
    record = record.T
    record.to_csv("records/record_test.csv", mode='w', header=False, index=False)
    print("Max score: ", np.max(score_record))
    print("Mean score: ", np.mean(score_record))
    print("Min score: ", np.min(score_record))
        
        
# Close enviroment
env.close()


