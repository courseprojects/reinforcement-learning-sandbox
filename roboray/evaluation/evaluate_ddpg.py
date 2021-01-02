"""

This module is used to evaluate the regular DDPG agent on multiple environments

"""
import os
from pathlib import Path

from roboray.agents.DDPG import DDPG
from roboray.utils.ddpg_utils import to_tensor
from roboray.utils.common_utils import load_config, set_logging
import gym

import torch


def train(agent, env, num_epochs, num_episodes, episode_horizon, warmup, render):
	log.info("Setting up Agent and Environment.")
	obs = env.reset()
	state_dim = env.observation_space.shape[0]
	iteration = 0
	log.info("Creating weights folders.")
	if not os.path.exists("weights"):
		os.mkdir("weights")
	log.info("Environment: {} \n Agent: {}\n".format(env.spec, agent.name))
	

	log.info("Commencing Warmup")
	while iteration <= warmup:
		obs = env.reset()
		agent.s_t = obs
		done = False
		while done==False:
			action = agent.random_action()
			obs, reward, done, info = env.step(action)
			agent.observe(reward, obs, done)
			iteration+=1

	log.info("Commencing Training")
	for epoch in range(num_epochs):
		epoch_reward = 0
		for episode in range(num_episodes):
			obs = env.reset()
			agent.s_t = obs
			done=False
			steps = 0
			while (done==False) & (steps<=episode_horizon): 
				action = agent.select_action(obs)     
				obs, reward, done, info = env.step(action)
				steps += 1
				epoch_reward+=reward
				agent.observe(reward, obs, done)
				agent.update_parameters()
		log.info("reward for epoch_{}: {}".format(epoch,epoch_reward/num_episodes))
		model = "./weights/model_{}.pt".format(epoch+1)
		torch.save(agent.actor, model)
		env_to_wrap = gym.make("InvertedDoublePendulum-v2")
		if render == True:
			render_rollout(env_to_wrap, model,epoch,record=False)


def render_rollout(env, model, epoch,record):
	if record == True:
		gym.wrappers.Monitor(env, model.format(epoch),force=True)
	actor = torch.load(model.format(epoch))
	actor.eval()
	obs = env.reset()
	done = False
	steps = 0
	while (done==False) & (steps <= 1000):
		env.render()
		obs, reward, done, info = env.step(actor(to_tensor(obs)).detach())

# Add rendering of all rollouts from training loop


if __name__=="__main__":
	# Getting path variables
	roboray_path = str(Path(os.getcwd()).parent)
	logging_path = roboray_path + "/config/default_logger.conf"
	double_pendulum_path = roboray_path + "/config/double_pendulum_ddpg.yaml" 

	# Set logging
	log = set_logging(logging_path)

	# Read config
	config = load_config(double_pendulum_path)
	

	# Train cartpole
	env = gym.make(config["environment"]["env_name"])
	agent = DDPG(**config["ddpg_agent"])
	train(agent, env, **config["training_params"])