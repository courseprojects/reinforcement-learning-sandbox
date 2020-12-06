# This script is currently a WIP.
from copy import deepcopy
import numpy as np
import robosuite as suite
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import ray
from roboray.train import train_ddpg

ray.init(address=auto)

if __name__=="__main__":
	train_ddpg(wandb_api=<api_key>, 
		wandb_project=<project_name>, 
		wandb_entity=<entity_name>,
		wandb_name=<run_name>)
