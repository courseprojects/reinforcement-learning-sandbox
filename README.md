# Reinforcement Learning Sandbox
<p float="left">
  <img src="assets/robotarmfull.png" width="300">
</p>
 <H1>Introduction</H1>
<p>
Welcome, in this project I train a multiple agents in different simulated environments. The main goal of this repository is to test applications of reinforcement learning to robot control problems but you will find other popular environments included for benchmarking. I leverage the simulation framework <a href="https://github.com/ARISE-Initiative/robosuite">robosuite</a> which is powered by <a href="http://mujoco.org/">MuJoCo</a> for the robot control problems. In future iteration I will consider moving to use the bullet physics engine.  
</p>
<p>
The original version of this repository was used as a submission for Stanford's course CS221 Autumn 2020. Subsequently this repository has been converted to be used for multiple different algorithm implementations:

- [x] REINFORCE
- [x] Deep Deterministic Policy Gradient
- [ ] Soft Actor Critic
- [ ] PPO
- [ ] Apex DDPG
</p>

<H1>Prerequisites</H1>
<p>
Currently, a <a href="http://mujoco.org/">MuJoCo</a> license is needed to create this simulation enviornment. For more information on getting a license please visit the this <a href="https://www.roboti.us/license.html">link</a>.
</p>

<H1>Setting up training enviornment</H1>
<p>
To simplify the creation of the training enviorment I have used Docker. You may find the Dockerfile used to build the container for the simulated environment in /Docker/Dockerfile. From here you will need to place your MuJoCo key in the same directory as the one you are building your docker container in. Given this is complete you may simply run the following command
</p>
<pre><code>$ docker build -t robosuite .
$ docker run -it -d --gpus all --name robosuite robosuite:latest
$ docker exec -t robosuite bash
</code></pre>
 <p>
 This will provide you with access to the Docker container within which you will have all of the necessary dependencies installed to get started (given you placed the MuJoCo key in the same directory your container was built). 
 </p>
<H1>Model Training</H1>
In order to track our model training I like to use <a href="https://www.wandb.com/">Weights and Biases</a>. Weights and biases enables efficient tracking of expriments and hyper parameter searches while storing all the results for you. You can find the old repo I used <a href="https://wandb.ai/peterdavidfagan/cs221-project">here</a>, I will be updating this to include a new repo soon.
  
<H1>Further</H1>
<p>
Feel free to check out our <a href="https://peterdavidfagan.gitbook.io/peter-david-fagan/robot-arm">docs</a> for further details.
</p>
