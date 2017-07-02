#! /usr/bin/python

# Gernerate expert data
import cPickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from dotmap import DotMap

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
            obs = env.reset()
            done = False
            totalreward = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None, :])
                observations.append(obs)
                actions.append(action)
                
                obs, reward, done, _ = env.step(action) 
                totalreward += reward
                steps += 1
                
                if render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
                returns.append(totalreward)
                
        expert_data = {'observations': np.array(observations),
               'actions': np.array(actions)}

        return expert_data
        
if __name__ == '__main__':
    expert_data = get_expert_data('experts/Hopper-v1.pkl', 'Hopper-v1', 
                             render=True, num_rollouts=20)
