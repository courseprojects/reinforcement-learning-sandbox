import wandb
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

os.environ['WANDB_API_KEY'] = ''
os.environ['WANDB_PROJECT'] = ''
os.environ['WANDB_NAME'] =  ''
os.environ['ENTITY'] = ''


if __name__ == "__main__":
	from models.REINFORCE import REINFORCE

	wandb.init()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	env = suite.make(
	    env_name="Lift",
	    robots="Panda",
	    has_renderer=False,
	    has_offscreen_renderer=False,
	    use_camera_obs=False,
	    use_object_obs=True,                    
	    horizon = 200, 
	    reward_shaping=True                 
	)
	obs = env.reset()
	state_dim = obs['robot0_robot-state'].shape[0]+obs['object-state'].shape[0]


	agent = REINFORCE(state_dim,env.action_dim)

	for episode in range(1000):
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
	        
	    agent.update_parameters(rewards, log_probs, 0.99)
	    print('Episode: {}, Rewards: {}'.format(episode, np.sum(rewards)))

