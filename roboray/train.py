import argparse, math, os, sys
from copy import deepcopy
import numpy as np
import robosuite as suite
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import ray
from roboray.models.REINFORCE.REINFORCE import REINFORCE
from roboray.models.DDPG.DDPG import DDPG


def create_env(env_name='Lift', robot='Panda',has_renderer=False,
                has_offscreen_renderer=False, use_camera_obs=False,
                use_object_obs=True, horizon=200, reward_shaping=True):

    env = suite.make(
        env_name=env_name,
        robots=robot,
        has_renderer=has_renderer,
        has_offscreen_renderer=has_offscreen_renderer,
        use_camera_obs=use_camera_obs,
        use_object_obs=use_object_obs,                    
        horizon = horizon, 
        reward_shaping=reward_shaping,
    )

    return env


def train_reinforce(env_name='Lift', robot='Panda',has_renderer=False,
                has_offscreen_renderer=False, use_camera_obs=False,
                use_object_obs=True, horizon=200, reward_shaping=True,
                gamma=0.99, lr=0.001, num_epochs=500, num_episodes=500, hidden_size=300, 
                **kwargs):
    '''
    This function trains our REINFORCE implementation.
    '''
    wandb_api = kwargs.get('wandb_api',None)
    wandb_project = kwargs.get('wandb_project',None)
    wandb_name =  kwargs.get('wandb_name',None)
    wandb_entity = kwargs.get('wandb_entity',None)

    env = create_env()
    obs = env.reset()
    state_dim = obs['robot0_robot-state'].shape[0]+obs['object-state'].shape[0]
    agent = REINFORCE(state_dim, env.action_dim, 
                    gamma, lr, num_episodes, 
                    horizon, hidden_size)
    for epoch in range(num_epochs):
        log_probs = [[] for i in range(num_episodes)]
        rewards = [[] for i in range(num_episodes)]
        for episode in range(num_episodes):
            obs=env.reset()
            done=False
            while done==False:
                state = torch.Tensor(np.append(obs['robot0_robot-state'],obs['object-state'])) 
                action, log_prob = agent.select_action(state)
                action_cpu = [x.to('cpu') for x in action]
                obs, reward, done, info = env.step(action_cpu)
                log_probs[episode].append(log_prob)
                rewards[episode].append(reward)
                
        agent.epoch_update_parameters(rewards, log_probs)
        print('Epoch: {}, Average_Rewards: {}'.format(epoch, np.sum(rewards,axis=1).mean()))
        if wandb.api is not None:
            wandb.log({'epoch_reward': np.sum(rewards,axis=1).mean()})

        if epoch%20==0:
            print('Saving model ...')
            torch.save(agent.model,'{}.pt'.format(wandb_name))
            if wandb.api is not None:
                wandb.save('{}.pt'.format(wandb_name))


def train_ddpg(env_name='Lift', robot='Panda',has_renderer=False,
                has_offscreen_renderer=False, use_camera_obs=False,
                use_object_obs=True, horizon=200, reward_shaping=True,
                gamma=0.99, lr_actor=0.001, lr_critic=0.001, tau=0.001,
                epsilon=10000, batch_size=64, max_mem_size=500000,enable_her=False,
                num_epochs=500 ,num_episodes=500, hidden_size=300, warmup=100 , **kwargs):
    '''
    This function trains our DDPG and DDPG+HER implementations.
    '''
    wandb_api = kwargs.get('wandb_api',None)
    wandb_project = kwargs.get('wandb_project',None)
    wandb_name =  kwargs.get('wandb_name',None)
    wandb_entity = kwargs.get('wandb_entity',None)
    if wandb_api is not None:
        wandb.init(config={'lr_actor':lr_actor})

    env = create_env()
    obs = env.reset()
    state_dim = obs['robot0_robot-state'].shape[0]+obs['object-state'].shape[0]
    agent = DDPG(state_dim, env.action_dim, env, hidden_size, 
                lr_actor, lr_critic, tau, gamma, epsilon, batch_size, 
                max_mem_size,enable_her)
    iteration = 0
    for epoch in range(num_epochs):
        rewards = [[] for i in range(num_episodes)]
        for episode in range(num_episodes):
            obs = env.reset()
            state = np.append(obs['robot0_robot-state'],obs['object-state'])
            agent.s_t = state
            done=False
            while done==False: 
                if iteration <= warmup:
                    action = agent.random_action()
                else:
                    action = agent.select_action(state)        
                iteration += 1

                obs, reward,done, info = env.step(action)
                rewards[episode].append(reward)
                state = np.append(obs['robot0_robot-state'],obs['object-state'])
                agent.observe(reward, state, done)
               
            # Update network weights after warm-up period
            if iteration > warmup:
                for _ in range(horizon):
                    agent.update_parameters()

        if reward_shaping==True:
            print('Epoch: {}, Average_Rewards: {}'.format(epoch, np.sum(rewards,axis=1).mean()))
            if wandb.api is not None:
                wandb.log({'epoch_reward': np.sum(rewards,axis=1).mean()})
        else:
            pass
            #implement tracking of sparse rewards here

        # Save models 
        if epoch%20==0:
            torch.save(agent.actor,'DDPG_{}.pt'.format(epoch))
            if wandb.api is not None:
                wandb.save('DDPG_{}.pt'.format(epoch))




# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='PyTorch robot arm training script')
#     parser.add_argument('--env_name', type=str, default='Lift')
#     parser.add_argument('--robot', type=str, default='Panda')
#     parser.add_argument('--algo', type=str, default='DDPG')
#     parser.add_argument('--enable_her',type=bool,default=False)
#     parser.add_argument('--dense_rewards',type=bool,default=True)
#     parser.add_argument('--hidden_size', type=int, default=256)
#     parser.add_argument('--max_mem_size', type=int, default=50000)
#     parser.add_argument('--tau', type=float, default=0.001)
#     parser.add_argument('--batch_size', type=int, default=64)
#     parser.add_argument('--lr', type=float, default=0.001)
#     parser.add_argument('--lr_actor', type=float, default=0.0001)
#     parser.add_argument('--lr_critic', type=float, default=0.001)
#     parser.add_argument('--epsilon', type=float, default=10000)
#     parser.add_argument('--warmup', type=int, default=100)
#     parser.add_argument('--theta', type=int, default=0.15)
#     parser.add_argument('--cube_x_distro',type=float, nargs='+', default=[0, 0],
#                         help='x distribution for cube location')
#     parser.add_argument('--cube_y_distro',type=float,  nargs='+', default=[0, 0],
#                         help='y distribution for cube location')
#     parser.add_argument('--enable_arm_randomization', type=bool, default=False,
#                         help='enable arm randomization (uniform) for each of the 7 joints')
#     parser.add_argument('--gamma', type=float, default=0.99,
#                         help='discount factor for reward (default: 0.99)')
#     parser.add_argument('--num_epochs',type=int, default=500,
#                         help='number of epochs to train on' )
#     parser.add_argument('--num_episodes', type=int, default=1,
#                         help='number of episodes per epoch')
#     parser.add_argument('--horizon', type=int, default=20,
#                         help='max episode length (default: 200)')
#     parser.add_argument('--wandb_api', type=str, default=None, 
#                         help='wandb api key')
#     parser.add_argument('--wandb_project', type=str, default='cs221-project',
#                         help='wandb project name')
#     parser.add_argument('--wandb_name', type=str, default='test-run',
#                         help='name of run')
#     parser.add_argument('--wandb_entity', type=str, default='peterdavidfagan', metavar='N',
#                         help='name of user running experiment')
    
#     args = parser.parse_args()
#     os.environ['WANDB_API_KEY'] = args.wandb_api
#     os.environ['WANDB_PROJECT'] = args.wandb_project
#     os.environ['WANDB_NAME'] =  args.wandb_name
#     os.environ['ENTITY'] = args.wandb_entity

#     if args.wandb_api is not None:
#         wandb.init(config=vars(args))
#     ray.init(address="auto")
    

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Setting up the robot enviornment
#     env = suite.make(
#         env_name=args.env_name,
#         robots=args.robot,
#         has_renderer=False,
#         has_offscreen_renderer=False,
#         use_camera_obs=False,
#         use_object_obs=True,                    
#         horizon = args.horizon, 
#         reward_shaping=args.dense_rewards,
#         placement_initializer = maybe_randomize_cube_location(args.cube_x_distro, args.cube_y_distro),
#         initialization_noise = maybe_randomize_robot_arm_location(args.enable_arm_randomization, 0.3)     
#     )
#     obs = env.reset()
#     state_dim = obs['robot0_robot-state'].shape[0]+obs['object-state'].shape[0]

#     # Setting algorithm according to args
#     if args.algo == 'REINFORCE':
#         print('Starting to train REINFORCE...')
#         train_reinforce()
#     elif args.algo == 'DDPG':
#         print('Starting to train DDPG')
#         train_ddpg()        
#     else:
#         sys.exit('Incorrect algorithms specification. Please check the algorithm argument provided.')

