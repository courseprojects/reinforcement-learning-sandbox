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
	# Initialize experiment tracking
	wandb.init()
