"""

This module is used to evaluate agents using the cartpole
gym environment.

"""
from roboray.agents.DDPG import DDPG
from roboray.utils.common_utils import load_config, set_logging
import gym


def train_cartpole(agent, env, num_epochs, num_episodes, episode_horizon, warmup, render):
	log.info("Setting up Agent and Environment.")
	obs = env.reset()
	state_dim = env.observation_space.shape[0]
	iteration = 0
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
			while (done==False) | (steps<=episode_horizon): 
				action = agent.action_high * agent.select_action(obs)  # Scale to match largest action      
				obs, reward, done, info = env.step(action)
				steps += 1
				epoch_reward+=reward
				agent.observe(reward, obs, done)
				agent.update_parameters()
		log.info("reward for epoch_{}: {}".format(epoch,epoch_reward/num_episodes))


	if render == True:
		obs = env.reset()
		for _ in range(1000):
			env.render()
			obs, reward, done, info = env.step(agent.select_action(obs))



if __name__=="__main__":
	# Set logging
	log = set_logging("/Users/peterfagan/Code/Roboray/roboray/config/default_logger.conf")

	# Read config
	config = load_config("/Users/peterfagan/Code/Roboray/roboray/config/cartpole_ddpg_default.yaml")
	

	# Train cartpole
	env = gym.make("InvertedPendulum-v2")
	agent = DDPG(**config["ddpg_agent"])
	train_cartpole(agent, env, **config["training_params"])

