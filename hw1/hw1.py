#! /usr/bin/python

# Gernerate expert data
import os, cPickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from dotmap import DotMap
import keras

def get_expert_data(expert_policy_file, envname, render=True, max_timesteps=None,
                    num_rollouts=20):
    """collect expert data
    Args:
        expert_policy_file: like /path/to/Hopper-v1.pkl
        envname: like Hopper-v1
    Returns:
        expert_data: dict with two keys, observations and actions
    """                
    
    policy_fn = load_policy.load_policy(expert_policy_file)
    
    with tf.Session():
        tf_util.initialize()
    
        # construct env
        env = gym.make(envname)
        max_steps = max_timesteps or env.spec.timestep_limit
        
        returns = []
        observations = []
        actions = []
        
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()   #(11, )
            done = False
            totalreward = 0.
            steps = 0
            while not done:
                #action (1,3)
                action = policy_fn(obs[None, :])
                observations.append(obs)
                actions.append(action[0])
                
                obs, reward, done, _ = env.step(action) 
                totalreward += reward
                steps += 1
                
                if render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
                returns.append(totalreward)
        
        # observations: (num, 11), actions:(num,3)  
        expert_data = {'observations': np.array(observations),
               'actions': np.array(actions)}

        return expert_data

def construct_model():
    from keras.layers import Dense, Activation
    tf.reset_default_graph()
    keras.backend.clear_session()
    model = keras.models.Sequential()
    model.add(Dense(units=128, input_dim=11))
    model.add(Activation('relu'))
    model.add(Dense(units=32))
    model.add(Activation('relu'))
    model.add(Dense(units=3))
    model.compile(loss='mean_squared_error', optimizer='adam', 
              metrics=['mean_squared_error'])
    return model

def evaludate(model, envname, render=True, max_timesteps=None,
              num_rollouts=20):
    """Evaludate the model 
    """
    print "Start evaludating."
        
    # construct env
    env = gym.make(envname)
    max_steps = max_timesteps or env.spec.timestep_limit
    
    returns = []
    observations = []
    actions = []
    
    for i in range(num_rollouts):
        print('iter', i)
        obs = env.reset()   #(11, )
        done = False
        totalreward = 0.
        steps = 0
        while not done:
            obs = obs[np.newaxis, :]
            action = model.predict(obs)             
            obs, reward, done, _ = env.step(action) 
            totalreward += reward
            steps += 1
            
            observations.append(obs)
            actions.append(action[0,:])
            
            if steps >= max_steps: break
            if render: env.render()
            
        returns.append(totalreward)
    return returns, {'observations': np.array(observations), 'actions': np.array(actions)}
    
def label_data(data, env_name):
    """ Label the data by expert, the data from evaluation session.
    """
    policy_fn = load_policy.load_policy('experts/'+task+'.pkl')
    batch_size = 128
    
    observations = data['observations']
    num = observations.shape[0]
    gt_labels = np.zeros_like(data['actions']) # (num, 3)
    
    with tf.Session():
        tf_util.initialize()
        for i in range(0, num, batch_size):
            gt_labels[i:i+batch_size] = policy_fn(
                observations[i:i+batch_size])
    return {'observations': observations, 'actions': gt_labels}
  
def dagger(origin_data, model, task, render, num_iters=1, num_rollouts=10):

    for i in range(num_iters):
        result, data = evaludate(model, task, render=render, num_rollouts=num_rollouts)
        dagger_data = label_data(data, task)
        
        new_data = np.concatenate((origin_data['observations'], dagger_data['observations']), axis=0)
        new_label = np.concatenate((origin_data['actions'], dagger_data['actions']), axis=0)
        hisory = model.fit(new_data, new_label, batch_size=512, epochs=10)
        
        origin_data['observations'] = new_data
        origin_data['actions'] = new_label
        print new_data.shape[0]
        
    return {'observations': new_data, 'actions': new_label}, model
      
if __name__ == '__main__':

    task = 'Hopper-v1'
    render = False
    num_rollouts = 10
    # Collect expert_data
    expert_data = {}
    if True:#not os.path.exists(task):
        expert_data = get_expert_data('experts/'+task+'.pkl', task, 
                                 render=render, num_rollouts=num_rollouts)
        with open(task, 'wb') as fo: cPickle.dump(expert_data, fo)
    else:
        with open(task, 'rb') as fi: expert_data = cPickle.load(fi)
    
    model = construct_model()
    history = model.fit(expert_data['observations'], expert_data['actions'],
                        batch_size=512, epochs=10)
                        
                        
    dagger(expert_data, model, task, render, num_iters=10, num_rollouts=num_rollouts)
    #result, data = evaludate(model, task, render=render, num_rollouts=40)
    #dagger_data = label_data(data, task)
    
    #new_data = np.concatenate((expert_data['observations'], dagger_data['observations']), axis=0)
    #new_label = np.concatenate((expert_data['actions'], dagger_data['actions']), axis=0)
    #hisory = model.fit(new_data, new_label, batch_size=512, epochs=10)
    evaludate(model, task, render=True, num_rollouts=5)
    
    
    
    
    
    
    
