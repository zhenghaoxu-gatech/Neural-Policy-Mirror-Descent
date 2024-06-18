# from typing import Optional
import torch
# import torch.nn as nn
import numpy as np
import gymnasium as gym
from torch.utils.data import TensorDataset, DataLoader, random_split
import torchvision.transforms as T
from model import ActorNet, CriticNet
from data import CriticDataset
from tqdm import tqdm
# import matplotlib.pyplot as plt
from envs.CartPoleEnvNoReset import get_cost
from datetime import datetime, timezone, timedelta
import time
from functools import partial
from itertools import chain
import os
import copy

from torch.multiprocessing import Lock, Pool
# import torch.multiprocessing as mp
from psutil import cpu_count
import sys
# from multiprocessing import cpu_count, Lock, freeze_support, Pool

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# resize = None   
# input 40 for 3x40x150; input 20 for 3x20x75

def get_screen_size(mode: str):
    if mode == "low":
        resize = T.Compose([T.ToPILImage(), T.Resize(20), T.ToTensor()])  
        return (3, 20, 75, resize)
    elif mode == "high":
        resize = T.Compose([T.ToPILImage(), T.Resize(40), T.ToTensor()])  
        return (3, 40, 150, resize)
    elif mode == "superhigh":
        resize = T.Compose([T.ToPILImage(), T.Resize(60), T.ToTensor()])  
        return (3, 60, 225, resize)
    raise "resolution unspecified!"

def get_batch_action(i, n_samples, batch_size):
    return i * batch_size // n_samples

def get_model_name(n_iters, resolution, gamma, sample_size, batch_size, epochs, lr, model_path, arch=0): 
    if arch == 0:
        actor_name = f'actor_{n_iters}_{resolution}_{gamma}_{sample_size}_{batch_size}_{epochs}_{lr}.pt'
        critic_name = f'critic_{n_iters}_{resolution}_{gamma}_{sample_size}_{batch_size}_{epochs}_{lr}.pt'
        result_name = f'result_{n_iters}_{resolution}_{gamma}_{sample_size}_{batch_size}_{epochs}_{lr}.npy'
    else:
        actor_name = f'actor_arch{arch}_{n_iters}_{resolution}_{gamma}_{sample_size}_{batch_size}_{epochs}_{lr}.pt'
        critic_name = f'critic_arch{arch}_{n_iters}_{resolution}_{gamma}_{sample_size}_{batch_size}_{epochs}_{lr}.pt'
        result_name = f'result_arch{arch}_{n_iters}_{resolution}_{gamma}_{sample_size}_{batch_size}_{epochs}_{lr}.npy'
    actor_name = os.path.join(model_path, actor_name)
    critic_name = os.path.join(model_path, critic_name)
    result_name = os.path.join(model_path, result_name)
    return actor_name, critic_name, result_name

def get_screen(env: gym.Env, resize):
    screen = np.array(env.render())
    # Cart is in the lower half, so strip off the top and bottom of the screen
    screen_height, screen_width, channel_size = screen.shape
    screen = screen[int(screen_height*0.4):int(screen_height * 0.8), :]
    screen = resize(screen)
    result = screen
    return 1-result # make white zeros

def get_init_screen(env: gym.Env, resize):
    screen = get_screen(env, resize)
    last_screen = torch.zeros_like(screen)
    return screen, last_screen

def step(env: gym.Env, action: int, policy: ActorNet, done: bool, last_screen: torch.Tensor, resize, mode="train"):
    reward = 0
    if done: 
        action = 2
    
    int_state, reward, terminated, truncated, _ = env.step(action)   # batch iterate
    done = done | terminated
    if mode == "eval":
        done = done | truncated

    # we use image difference to capture velocities.
    screen = get_screen(env, resize)        # s_{t+1}
    s_next = screen - last_screen
    a_next = policy.get_action(s_next)[0]    # a_{t+1}
    last_screen_next = screen if not done else last_screen

    return reward, s_next, a_next, done, last_screen_next

def get_mask(worker_id):
    if sys.platform == 'linux':
        mask = {worker_id * 2}
    else:
        mask = 1
        mask <<= (worker_id*2)
    return mask

def sample_async(env_name: str, sample_policy: ActorNet, sample_action: int, gamma: float, resize, action_list: None):
    env = gym.make(env_name, render_mode="rgb_array")
    id, action_list = action_list
    if action_list is None:
        action_list = [sample_action]

    pid = os.getpid()
    # pin each subprocess on a physical core
    if sys.platform == 'linux':
        pass
        # os.sched_setaffinity(pid, get_mask(id))
    else:     
        pass
        # import affinity
        # affinity.set_process_affinity_mask(pid, get_mask(id))   
              
    results = []
    progress_bar = tqdm(total=len(action_list), position=id, desc=f"Worker {id}", leave=False)
    for i, sample_action in enumerate(action_list):
        # print(f'process {id} begin sampling action {sample_action}.')
        env.reset()

        state, last_screen = get_init_screen(env, resize)             # s_0 
        action = sample_policy.get_action(state)[0]  # a_0

        done = False
        while not done:
            p = np.random.uniform(0.0, 1.0)
            if p <= gamma:
                reward, state, action, done, last_screen = step(env, action, sample_policy, done, last_screen, resize)
            else:
                break
            
        sout = state
        # (s_t), no cost yet
        reward, state, action, done, last_screen = step(env, sample_action, sample_policy, done, last_screen, resize)
        # (s_t, a_sample, c_t, s_{t+1})
        cost = get_cost(reward, gamma)

        while not done:
            p = np.random.uniform(0.0, 1.0)
            if p <= gamma:
                reward, state, action, done, last_screen = step(env, action, sample_policy, done, last_screen, resize)
            else:
                break
        
        reward, state, action, done, last_screen = step(env, action, sample_policy, done, last_screen, resize)
        # (s_h, a_h, c_h)
        cost_prime = get_cost(reward, gamma)
        
        results.append((sout, cost, cost_prime))

        progress_bar.update(1)
    
    if len(results) == 0:
        results = results[0]

    return results
    
def generate_samples_async(env_name: str, sample_policy: ActorNet, n_samples: int, gamma: float, resize, n_envs=1):
    assert n_samples % n_envs == 0, "number of samples is not standard!"
    cpus = cpu_count(logical=False)
    assert n_envs <= cpus
    print('number of environments: {} \t number of cpus: {}'.format(n_envs, cpus))
    n_actions = 2
    n_samples_per_env = n_samples // n_envs
    
    actions = []
    for a in range(n_actions):
        actions += [a] * n_samples
        
    action_list = np.array_split(actions, n_envs)
    
    states = []
    values = []
    results = None
    sample_partial = partial(sample_async, env_name, sample_policy, 0, gamma, resize)
    if n_envs > 1:
        # print(mp.get_start_method())
        with Pool(processes=n_envs, initializer=tqdm.set_lock, initargs=(Lock(),), maxtasksperchild=1) as pool:
            # start_time = time.time()
            result_list = pool.map(sample_partial, list(zip(range(n_envs), action_list)))
            # print("--- %s seconds ---" % (time.time() - start_time))
    else:
        # start_time = time.time()
        result_list = list(map(sample_partial, zip(range(n_envs), action_list)))
        # print("--- %s seconds ---" % (time.time() - start_time))

    results = list(chain(*result_list))
    for result in results:
        # state, cost, cost_prime = result.get()
        state, cost, cost_prime = result
        states.append(state)
        values.append(cost + cost_prime * gamma / (1-gamma))


    states = torch.stack(states)
    actions = torch.tensor(actions)
    values = torch.tensor(values)
    # print(states.size(), actions.size(), values.size())
    return states, actions, values

def make_critic_data(states, actions, values, batch_size, test_batch=1):   # data=(s, a, Q'(s,a))
    n_samples = states.size()[0]
    assert n_samples % batch_size == 0
    critic_dataset = CriticDataset(states, actions, values)
    test_size = test_batch * batch_size
    train_size = n_samples - test_size
    train_dataset, test_dataset = random_split(critic_dataset, 
                                               [train_size, test_size]) 
    critic_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                                   shuffle=True) 
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, 
                                 shuffle=True)
    return critic_dataloader, test_dataloader

def make_actor_data(critic_data: DataLoader, critic_val: DataLoader, actor: ActorNet, critic: CriticNet, batch_size, iteration, test_batch=1):
    actor_data_x = None
    actor_data_y = None
    for i, batch in enumerate(critic_data):
        batch_states, batch_actions, batch_target_values = batch
        batch_states = batch_states.to(DEVICE)
        pred_critic = critic.get_values(batch_states)
        pred_actor = actor(batch_states)
        if iteration == 0:
            target = -pred_critic
        else:
            target = actor.gamma * pred_actor - pred_critic
        if actor_data_x is None:
            actor_data_x = batch_states
            actor_data_y = target
        else:
            actor_data_x = torch.cat([actor_data_x, batch_states])
            actor_data_y = torch.cat([actor_data_y, target])

    for i, batch in enumerate(critic_val):
        batch_states, batch_actions, batch_target_values = batch
        batch_states = batch_states.to(DEVICE)
        pred_critic = critic.get_values(batch_states)
        pred_actor = actor(batch_states)
        if iteration == 0:
            target = -pred_critic
        else:
            target = actor.gamma * pred_actor - pred_critic
        actor_data_x = torch.cat([actor_data_x, batch_states])
        actor_data_y = torch.cat([actor_data_y, target])
        
    actor_dataset = TensorDataset(actor_data_x, actor_data_y)
    test_size = test_batch * batch_size
    train_size = len(actor_dataset) - test_size
    train_dataset, test_dataset = random_split(actor_dataset, 
                                               [train_size, test_size]) 
    actor_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True) 
    test_dataloader = DataLoader(test_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True) 
    
    return actor_dataloader, test_dataloader

def make_actor_data_combined(critic_data: DataLoader, critic_val: DataLoader, actor: ActorNet, batch_size, iteration, test_batch=1):
    actor_data_x = None
    actor_data_a = None
    actor_data_y = None
    for i, batch in enumerate(critic_data):
        batch_states, batch_actions, batch_target_values = batch
        batch_states = batch_states.to(DEVICE)
        pred_actor = actor.get_score(batch_states, batch_actions)
        if iteration == 0:
            target = -batch_target_values
        else:
            target = actor.gamma * pred_actor - batch_target_values
        if actor_data_x is None:
            actor_data_x = batch_states
            actor_data_a = batch_actions
            actor_data_y = target
        else:
            actor_data_x = torch.cat([actor_data_x, batch_states])
            actor_data_a = torch.cat([actor_data_a, batch_actions])
            actor_data_y = torch.cat([actor_data_y, target])

    for i, batch in enumerate(critic_val):
        batch_states, batch_actions, batch_target_values = batch
        batch_states = batch_states.to(DEVICE)
        pred_actor = actor.get_score(batch_states, batch_actions)
        if iteration == 0:
            target = -batch_target_values
        else:
            target = actor.gamma * pred_actor - batch_target_values
        actor_data_x = torch.cat([actor_data_x, batch_states])
        actor_data_a = torch.cat([actor_data_a, batch_actions])
        actor_data_y = torch.cat([actor_data_y, target])
        
    actor_dataset = CriticDataset(actor_data_x, actor_data_a, actor_data_y)
    test_size = test_batch * batch_size
    train_size = len(actor_dataset) - test_size
    train_dataset, test_dataset = random_split(actor_dataset, 
                                               [train_size, test_size]) 
    actor_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True) 
    test_dataloader = DataLoader(test_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True) 
    
    return actor_dataloader, test_dataloader

def get_reward(env_name: str, actor: ActorNet, resize, id: int):
    env = gym.make(env_name, render_mode="rgb_array")
    env.reset()

    state, last_screen = get_init_screen(env, resize)             # s_0 
    action = actor.get_action(state)[0]  # a_0

    tot_reward = 0
    done = False
    while (not done) and (tot_reward < 200):
        reward, state, action, done, last_screen = step(env, action, actor, done, last_screen, resize, mode='eval')
        tot_reward = tot_reward + reward
        
    return tot_reward

def eval_actor(env_name: str, actor: ActorNet, resize, eval_size=16, n_envs=1):

    print("current temperature: ", actor.temperature)
    actor.eval()
    tot_rewards = []
    get_reward_partial = partial(get_reward, env_name, actor, resize)
    if n_envs > 1:
        with Pool(processes=n_envs) as pool:
            results = pool.map(get_reward_partial, range(eval_size))
    else:
        results = list(map(get_reward_partial, range(eval_size)))
                        # get_reward, (env_name, actor)) for k in range(n_envs)]
    for result in results:
        tot_rewards.append(result)
    return np.array(tot_rewards)
    
