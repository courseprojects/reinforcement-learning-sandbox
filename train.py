import argparse, math, os
import robosuite as suite
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.REINFORCE import REINFORCE

parser = argparse.ArgumentParser(description='PyTorch robot arm training script')
parser.add_argument('--env_name', type=str, default='Lift')
parser.add_argument('--robot', type=str, default='Panda')
parser.add_argument('--algo', type=str, default='REINFORCE')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--num_episodes', type=int, default=500, metavar='N',
                    help='number of episodes')
parser.add_argument('--horizon', type=int, default=200, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--wandb_api', type=str, default=None, metavar='N',
                    help='wandb api key')
parser.add_argument('--wandb_project', type=str, default='cs221-project', metavar='N',
                    help='wandb project name')
parser.add_argument('--wandb_name', type=str, default='test-run', metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--wandb_entity', type=str, default='peterdavidfagan', metavar='N',
                    help='user running experiment')
args = parser.parse_args()

os.environ['WANDB_API_KEY'] = args.wandb_api
os.environ['WANDB_PROJECT'] = args.wandb_project
os.environ['WANDB_NAME'] =  args.wandb_name
os.environ['ENTITY'] = args.wandb_entity


wandb.init()
wandb.config.gamma = args.gamma
wandb.config.horizon = args.horizon
wandb.config.num_episodes = args.num_episodes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


env = suite.make(
    env_name=args.env_name,
    robots=args.robot,
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    use_object_obs=True,                    
    horizon = args.horizon, 
    reward_shaping=True                 
)
obs = env.reset()
state_dim = obs['robot0_robot-state'].shape[0]+obs['object-state'].shape[0]

if args.algo=='REINFORCE':
	agent = REINFORCE(state_dim,env.action_dim)
else:
	print('Incorrect algorithms specification. Please check the algorithm argument provided.')

for episode in range(args.num_episodes):
    obs=env.reset()
    state = torch.Tensor(np.append(obs['robot0_robot-state'],obs['object-state']))
    done=False
    log_probs = []
    rewards = []
    while done==False: 
        action, log_prob = agent.select_action(state)
        obs, reward, done, info = env.step(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        
    agent.update_parameters(rewards, log_probs, args.gamma)
    print('Episode: {}, Rewards: {}'.format(episode, np.sum(rewards)))
    wandb.log({'episode_reward': np.sum(rewards)})

torch.save(agent.state_dict(),'./{}.pkl'.format(args.wandb_name))
wandb.save('{}.pkl'.format(args.wandb_name))
