import argparse, math, os, sys
import numpy as np
import robosuite as suite
import wandb
import imageio
import plotly.graph_objects as go

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.REINFORCE import REINFORCE
from models.DDPG import DDPG


def set_agent(state_dim, env, args):
    # Setting algorithm according to args
    if args.algo=='REINFORCE':
        agent = REINFORCE(state_dim,env.action_dim, args.gamma, args.lr, args.num_episodes, args.horizon, args.hidden_size)
        agent.model = torch.load(args.model_path) 
        return agent
    elif args.algo=='DDPG':
        agent = DDPG(state_dim, env.action_dim, env, args)
        agent.actor = torch.load(args.model_path) 
        return agent
    else:
        sys.exit('Incorrect algorithms specification. Please check the algorithm argument provided.')


def create_media():
    '''This function is taken from the robosuite repository demo folder.'''
    # initialize an environment with offscreen renderer
    # requires kaleido
    env = suite.make(
        args.env_name,
        args.robot,
        has_renderer=False,
        ignore_done=True,
        use_camera_obs=True,
        use_object_obs=True,
        horizon = args.horizon, 
        camera_names=args.camera,
        camera_heights=args.height,
        camera_widths=args.width,
    )
    obs = env.reset()
    state_dim = obs['robot0_robot-state'].shape[0]+obs['object-state'].shape[0]
    state = np.append(obs['robot0_robot-state'],obs['object-state'])
    agent = set_agent(state_dim, env, args)


    # create a video writer with imageio
    writer = imageio.get_writer(args.video_path, fps=20)

    frames = []
    trajectory = []
    cube_pos = []
    for i in range(args.horizon):
        if args.algo=='REINFORCE':
            action, log_prob = agent.select_action(state)
        else:
            action = agent.select_action(state)  
        cube_pos.append(env.sim.data.body_xpos[env.cube_body_id])
        trajectory.append(list(env.sim.data.site_xpos[env.robots[0].eef_site_id]))
        obs, reward, done, info = env.step(action)
        state = np.append(obs['robot0_robot-state'],obs['object-state'])
        # dump a frame from every K frames
        if i % args.skip_frame == 0:
            frame = obs[args.camera + "_image"][::-1]
            writer.append_data(frame)
            print("Saving frame #{}".format(i))

        if done:
            break
    fig=go.Figure()
    fig.add_trace(go.Scatter3d(x=[x[0] for x in trajectory],y=[x[1] for x in trajectory],z=[x[2] for x in trajectory]))
    fig.add_trace(go.Scatter3d(x=[cube_pos[0][0]],y=[cube_pos[0][1]],z=[cube_pos[0][2]],
                                marker=dict(
                                size=20,
                                symbol='square',
                            )))
    fig.update_layout()
    fig.show()
    fig.write_image('plot.png')
    writer.close()

def watch_trajectory():
    # Setting up the robot enviornment
    env = suite.make(
        env_name=args.env_name,
        robots=args.robot,
        has_renderer=args.render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,                    
        horizon = args.horizon, 
        reward_shaping=False                
    )
    obs = env.reset()
    state_dim = obs['robot0_robot-state'].shape[0]+obs['object-state'].shape[0]
    state = np.append(obs['robot0_robot-state'],obs['object-state'])
    agent = set_agent(state_dim, env, args)

    # Visualize a single run
    done=False
    while done==False: 
        if args.algo=='REINFORCE':
            action, log_prob = agent.select_action(state)
        else:
            action = agent.select_action(state)           
        obs, reward, done, info = env.step(action)
        state = np.append(obs['robot0_robot-state'],obs['object-state'])
        env.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch robot arm testing script')
    parser = argparse.ArgumentParser(description='PyTorch robot arm training script')
    parser.add_argument('--env_name', type=str, default='Lift')
    parser.add_argument('--robot', type=str, default='Panda')
    parser.add_argument('--algo', type=str, default='DDPG')
    parser.add_argument('--model_path', type=str, default='Lift')
    parser.add_argument('--render', type=bool, default=True)
    parser.add_argument("--camera", type=str, default="frontview", help="Name of camera to render")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--video_path", type=str, default="video.mp4")
    parser.add_argument("--skip_frame", type=int, default=1)
    parser.add_argument('--enable_her',type=bool,default=False)
    parser.add_argument('--dense_rewards',type=bool,default=True)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--max_mem_size', type=int, default=50000)
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_actor', type=float, default=0.0001)
    parser.add_argument('--lr_critic', type=float, default=0.001)
    parser.add_argument('--epsilon', type=float, default=10000)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--theta', type=int, default=0.15)
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--horizon', type=int, default=200,
                        help='max episode length (default: 200)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #watch_trajectory()
    create_media()
