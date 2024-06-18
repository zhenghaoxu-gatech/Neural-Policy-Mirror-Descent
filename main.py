import torch
import torch.nn as nn

import numpy as np
from NPMD_utils import get_screen_size, generate_samples_async, make_critic_data, make_actor_data, eval_actor, get_model_name, make_actor_data_combined
from model import ActorNet, CriticNet
from tqdm import tqdm
from copy import deepcopy
# from envs.CartPoleEnvNoReset import CartPoleEnvNoReset
# from multiprocessing import set_start_method
import argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def NPMD(n_iters, resolution, gamma, sample_size, batch_size, epochs, lr, env, n_actions, eval_size, n_envs, model_path, resume=0, arch=0):


    # set_start_method("spawn")

    C, H, W, resize = get_screen_size(resolution)
    actor_net = ActorNet(H, W, n_actions, gamma, arch=arch)
    critic_net = CriticNet(H, W, n_actions, arch=arch)
    trained_reward = []
    
    actor_name, critic_name, result_name = get_model_name(n_iters, 
                                                          resolution, 
                                                          gamma, 
                                                          sample_size, 
                                                          batch_size, 
                                                          epochs, 
                                                          lr, 
                                                          model_path, 
                                                          arch=arch)
    print(result_name)
    if resume > 0: 
        critic_net.load_state_dict(torch.load(critic_name))
        actor_net.load_state_dict(torch.load(actor_name))
        actor_net.temperature = actor_net.gamma ** resume
        trained_reward = np.load(result_name).tolist()
    critic_net.to(DEVICE)

    print('=== initial actor performance ===')
    actor_net.eval()
    with torch.no_grad():
        rewards = eval_actor(env, actor_net, resize, eval_size=eval_size)
    print('averaged reward: {} \t rewards: {}'.format(rewards.mean(), rewards))
    if resume > 0:  # re-evaluate last performance
        trained_reward[-1] = rewards.mean()
    print(actor_net.gamma, actor_net.temperature)

    # exit()

    for iter in tqdm(range(resume, n_iters)):  # NPMD iterations
        print(f'========== iteration {iter} begin ==========')
        print('sampling...')
        
        actor_net.to('cpu')
        actor_net.eval()
        with torch.no_grad():
            states, actions, values = generate_samples_async(env, actor_net, sample_size, gamma, resize, n_envs=n_envs)

        actor_net.to(DEVICE)

        critic_data, critic_val = make_critic_data(states, actions, values, batch_size)
        print(len(critic_data), len(critic_val))
        print('sampling finished.')

        print('critic training...')

        optimizer_critic = torch.optim.SGD(
            critic_net.parameters(), 
            lr=lr)

        mse_loss = nn.MSELoss()

        best_critic_loss = np.inf
        best_critic = None
        for epoch in tqdm(range(epochs), leave=False):
            loss_train = []
            critic_net.train()
            for i, batch in enumerate(critic_data): 
                batch_states, batch_actions, batch_target_values = batch
                batch_states = batch_states.to(DEVICE)
                batch_actions = batch_actions.to(DEVICE)
                batch_target_values = batch_target_values.to(DEVICE)
                pred_Q = critic_net(batch_states, batch_actions)

                optimizer_critic.zero_grad()
                loss = mse_loss(pred_Q, batch_target_values)
                loss.backward()
                optimizer_critic.step()
                loss_train.append(loss.item())

            
            # # validation
            loss_val = []
            critic_net.eval()
            with torch.no_grad():
                for i, batch in enumerate(critic_val):
                    batch_states, batch_actions, batch_target_values = batch
                    batch_states = batch_states.to(DEVICE)
                    batch_actions = batch_actions.to(DEVICE)
                    batch_target_values = batch_target_values.to(DEVICE)
                    pred_Q = critic_net(batch_states, batch_actions)
                    loss = mse_loss(pred_Q, batch_target_values)

                    loss_val.append(loss.item())

            if np.mean(loss_val) < best_critic_loss:
                best_critic_loss = np.mean(loss_val)
                best_critic = deepcopy(critic_net.state_dict())
            
            if epoch + 1 == epochs:
                critic_training_res = 'train loss: {:.7f} \t validate loss: {:.7f} \t best loss: {:.7f}'.format(
                    np.mean(loss_train), 
                    np.mean(loss_val), 
                    best_critic_loss)

        critic_net.load_state_dict(best_critic)
        print('critic training finished.', critic_training_res)

        critic_net.eval()
        with torch.no_grad():
            actor_data, actor_val = make_actor_data(critic_data, critic_val, actor_net, critic_net, batch_size, iter)

        print('actor training...')
        optimizer_actor = torch.optim.SGD(
            actor_net.parameters(), 
            lr=lr)
        
        best_actor_loss = np.inf
        best_actor = None
        for epoch in tqdm(range(epochs), leave=False): 
            actor_net.train()
            loss_train = []
            for i, batch in enumerate(actor_data): 
                batch_states, batch_targets = batch
                batch_states, batch_targets = batch_states.to(DEVICE), batch_targets.to(DEVICE)
                preds = actor_net(batch_states)

                optimizer_actor.zero_grad()
                loss = mse_loss(preds, batch_targets)
                loss.backward()
                optimizer_actor.step()
                loss_train.append(loss.item())
            
            loss_val = []
            actor_net.eval()
            with torch.no_grad():
                for i, batch in enumerate(actor_val):
                    batch_states, batch_targets = batch
                    batch_states, batch_targets = batch_states.to(DEVICE), batch_targets.to(DEVICE)
                    preds = actor_net(batch_states)
                    loss = mse_loss(preds, batch_targets)
                    loss_val.append(loss.item())
                
            if np.mean(loss_val) < best_actor_loss:
                best_actor_loss = np.mean(loss_val)
                best_actor = deepcopy(actor_net.state_dict())
                
            if epoch + 1 == epochs:
                actor_training_res='train loss: {:.7f} \t validate loss: {:.7f} \t best loss: {:.7f}'.format(
                    np.mean(loss_train), 
                    np.mean(loss_val), 
                    best_actor_loss)

        actor_net.load_state_dict(best_actor)
        actor_net.step_temperature()        # update temperature
        print('actor training finished.', actor_training_res)

        print('=== actor performance ===')
        actor_net.to('cpu')
        actor_net.eval()
        with torch.no_grad():
            rewards = eval_actor(env, actor_net, resize, eval_size=eval_size, n_envs=n_envs)
        print('averaged reward: {} \t rewards: {}'.format(rewards.mean(), rewards))
        trained_reward.append(rewards.mean())

        torch.save(critic_net.state_dict(), critic_name)
        torch.save(actor_net.state_dict(), actor_name)
        np.save(result_name, np.array(trained_reward))
        print(f'========== iteration {iter} end ==========')

    print(trained_reward)
    
def NPMD_combined(n_iters, resolution, gamma, sample_size, batch_size, epochs, lr, env, n_actions, eval_size, n_envs, model_path):
    C, H, W, resize = get_screen_size(resolution)
    actor_net = ActorNet(H, W, n_actions, gamma)
    
    actor_name, critic_name, result_name = get_model_name(n_iters, 
                                                          resolution, 
                                                          gamma, 
                                                          sample_size, 
                                                          batch_size, 
                                                          epochs, 
                                                          lr, 
                                                          model_path)

    print('=== initial actor performance ===')
    actor_net.eval()
    with torch.no_grad():
        rewards = eval_actor(env, actor_net, resize, eval_size=eval_size)
    print('averaged reward: {} \t rewards: {}'.format(rewards.mean(), rewards))
    print(actor_net.gamma, actor_net.temperature)

    # exit()

    trained_reward = []
    for iter in tqdm(range(n_iters)):  # NPMD iterations
        print(f'========== iteration {iter} begin ==========')
        print('sampling...')
        
        actor_net.to('cpu')
        actor_net.eval()
        states, actions, values = generate_samples_async(env, actor_net, sample_size, gamma, resize, n_envs=n_envs)
        actor_net.to(DEVICE)


        critic_data, critic_val = make_critic_data(states, actions, values, batch_size)
        print(len(critic_data), len(critic_val))
        print('sampling finished.')

        print('actor training...')
        with torch.no_grad():
            actor_data, actor_val = make_actor_data_combined(critic_data, critic_val, actor_net, batch_size, iter)

        optimizer_actor = torch.optim.SGD(
            actor_net.parameters(), 
            lr=lr)
        mse_loss = nn.MSELoss()
        best_actor_loss = np.inf
        for epoch in tqdm(range(epochs), leave=False): 
            actor_net.train()
            loss_train = []
            for i, batch in enumerate(actor_data): 
                batch_states, batch_actions, batch_targets = batch
                batch_states, batch_actions, batch_targets = batch_states.to(DEVICE), batch_actions.to(DEVICE), batch_targets.to(DEVICE)
                preds = actor_net.get_score(batch_states, batch_actions)

                optimizer_actor.zero_grad()
                loss = mse_loss(preds, batch_targets)
                loss.backward()
                optimizer_actor.step()
                loss_train.append(loss.item())
            
            loss_val = []
            actor_net.eval()
            with torch.no_grad():
                for i, batch in enumerate(actor_val):
                    batch_states, batch_actions, batch_targets = batch
                    batch_states, batch_actions, batch_targets = batch_states.to(DEVICE), batch_actions.to(DEVICE), batch_targets.to(DEVICE)
                    preds = actor_net.get_score(batch_states, batch_actions)
                    loss = mse_loss(preds, batch_targets)
                    loss_val.append(loss.item())
                
            if np.mean(loss_val) < best_actor_loss:
                torch.save(actor_net.state_dict(), actor_name)
                best_actor_loss = np.mean(loss_val)
                
            if epoch + 1 == epochs:
                actor_training_res='train loss: {:.7f} \t validate loss: {:.7f} \t best loss: {:.7f}'.format(
                    np.mean(loss_train), 
                    np.mean(loss_val), 
                    best_actor_loss)

        actor_net.load_state_dict(torch.load(actor_name))
        actor_net.step_temperature()        # update temperature
        print('actor training finished.', actor_training_res)

        print('=== actor performance ===')
        actor_net.to('cpu')
        actor_net.eval()
        with torch.no_grad():
            rewards = eval_actor(env, actor_net, resize, eval_size=eval_size, n_envs=n_envs)
        print('averaged reward: {} \t rewards: {}'.format(rewards.mean(), rewards))
        trained_reward.append(rewards.mean())


        np.save(result_name, np.array(trained_reward))
        print(f'========== iteration {iter} end ==========')

    print(trained_reward)

if __name__ == '__main__':    
    torch.multiprocessing.set_start_method('spawn')
    
    parser = argparse.ArgumentParser(description="Neural Policy Mirror Descent")

    # Positional Argument
    parser.add_argument("n_iters", type=int, help="Number of NPMD iterations", default=200)
    parser.add_argument("resolution", choices=["high", "low", "superhigh"], help="Image resolution option", default="high")
    parser.add_argument("gamma", type=float, help="Discount factor", default=0.98)
    parser.add_argument("sample_size", type=int, help="Number of samples per action", default=1024)
    parser.add_argument("batch_size", type=int, help="SGD batch size", default=256)
    parser.add_argument("epochs", type=int, help="SGD epochs", default=200)
    parser.add_argument("lr", type=float, help="SGD learning rate", default=0.001)

    # Optional Arguments
    parser.add_argument("--env", help="Environment name", default='CartPole-noreset')
    parser.add_argument("--model_path", help="Save path", default="./results/0/")
    parser.add_argument("--n_actions", type=int, help="Number of actions", default=2)
    parser.add_argument("--eval_size", type=int, help="Number of performannce evaluation environments", default=32)
    parser.add_argument("--n_envs", type=int, help="Number of parallel sampling workers", default=1)
    parser.add_argument("--combine", action='store_true', help="Combine critic and actor updates", default=False)
    parser.add_argument("--resume", type=int, help="Resume from iteration", default=0)
    parser.add_argument("--arch", type=int, help="Different architectures for high-resolution images", default=0)

    args = parser.parse_args()

    if args.combine:
        NPMD_combined(n_iters=args.n_iters, 
         resolution=args.resolution,
         gamma=args.gamma,
         sample_size=args.sample_size, 
         batch_size=args.batch_size, 
         epochs=args.epochs, 
         lr=args.lr, 
         env=args.env, 
         n_actions=args.n_actions,
         eval_size=args.eval_size,
         n_envs=args.n_envs,
         model_path=args.model_path
         )
    else:
        NPMD(n_iters=args.n_iters, 
            resolution=args.resolution,
            gamma=args.gamma,
            sample_size=args.sample_size, 
            batch_size=args.batch_size, 
            epochs=args.epochs, 
            lr=args.lr, 
            env=args.env, 
            n_actions=args.n_actions,
            eval_size=args.eval_size,
            n_envs=args.n_envs,
            model_path=args.model_path,
            resume=args.resume,
            arch=args.arch
            )
