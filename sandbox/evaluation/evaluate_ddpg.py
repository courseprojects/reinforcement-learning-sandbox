"""

This module is used to evaluate the regular DDPG agent on multiple environments

"""
import os
from pathlib import Path

from sandbox.agents.DDPG import DDPG
from sandbox.utils.ddpg_utils import to_tensor
from sandbox.utils.common_utils import load_config, set_logging
from sandbox.env.robosuite_lift import GymWrapper

import robosuite as suite
import gym

import torch


def train(agent, env, num_epochs, num_episodes, episode_horizon, warmup, render):
	"""
	This function trains a ddpg agent given training parameters provided by config file.
	"""
	log.info("Setting up Agent and Environment.")
	obs = env.reset()
	iteration = 0
	log.info("Creating weights folders.")
	if not os.path.exists("weights"):
		os.mkdir("weights")

	log.info("Environment: {} \n Agent: {}\n".format(config["environment"]["env_name"], agent.name))
	

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
		if config["environment"]["env_source"]=="openai_gym":		
			env_to_wrap = gym.make(config["environment"]["env_name"])
		else:
			env_to_wrap = env
		if render == True:
			render_rollout(env_to_wrap, model,epoch,record=False)


def render_rollout(env, model, epoch,record):
	"""
	This function renders a rollout given a certain set of model weights.
	"""
	if (record == True) & (config["environment"]["env_source"]=="openai_gym"):
		gym.wrappers.Monitor(env, model.format(epoch),force=True)
	actor = torch.load(model.format(epoch))
	actor.eval()
	obs = env.reset()
	done = False
	steps = 0
	while (done==False) & (steps <= config["training_params"]["episode_horizon"]):
		env.render()
		action = actor(to_tensor(obs)).detach().numpy()
		obs, reward, done, info = env.step(action)
	env.close()



if __name__=="__main__":
	# Getting path variables
	sandbox_path = str(Path(os.getcwd()).parent)
	logging_path = sandbox_path + "/config/default_logger.conf"
	config_path = sandbox_path + "/config/lift_ddpg.yaml" 

	# Set logging
	log = set_logging(logging_path)

	# Read config
	config = load_config(config_path)
	
	# Train cartpole
	if config["environment"]["env_source"]=="openai_gym":
		env = gym.make(config["environment"]["env_name"])
	elif config["environment"]["env_source"]=="stanford_robosuite":
		env = suite.make(**config["robosuite_env"])
		env = GymWrapper(env, keys=config["robosuite_keys"])
	else:
		log.info("Error setting the enviroment.")


	agent = DDPG(**config["ddpg_agent"])
	train(agent, env, **config["training_params"])