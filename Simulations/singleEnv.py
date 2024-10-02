from Assets.predator import Predator
from Assets.prey import Prey

import pygame, sys, copy, math
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO, A2C
from gymnasium.envs.registration import register
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import random
import matplotlib.pyplot as plt
import cv2

class SimWolf(gym.Env):
    """Custom environment designed for predator-prey simulation that follows gym interface."""
    metadata = {"render_modes": ["human"], "render_fps": 30}
    """Intialize simulation constants"""
    WIN_WIDTH = 1000
    WIN_HEIGHT = 750
    DIRECTIONS = [1, 0, -1]
    
    def __init__(self, render_mode="human"):
        super().__init__()
        """ Define action and observation space
            Actions:    possible movements for the AI, in this case, 360 actions for rotation, 
                        discrete action means only one action at a time, as opposed to 
                        continuous action in our case to make 5 actions each time (5 predators)
                        
            Observations:   factors that impact the decision of what action to make, in this case, 
                            possition of other predators, and position of prey (5 observations)
        """
        self.action_space = spaces.Box(low=-1, high =1, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(4,), dtype=np.float64)
        
        """initialize variables needed to move the predator and prey"""
        self.current_pred_angle = 0
        self.wolves = []
        self.predator_poses = []
        self.prev_predator_poses = []
        for i in range(0, 1):
            wolfX, wolfY = (random.randrange(0, 1000), random.randrange(0, 750))
            self.wolves.append(Predator(wolfX, wolfY))
            self.predator_poses.append((wolfX, wolfY))
            self.prev_predator_poses.append((wolfX, wolfY))
            
        self.current_prey_angle = 0
        preyX, preyY = (random.randrange(0, 1000), random.randrange(0, 750))
        self.prey = Prey(preyX, preyY)
        self.prey_pos = (preyX, preyY)
        self.prev_prey_pos = (preyX, preyY)
        
        """Initialize variables needed for counting steps and episodes (miscellaneous purposes)"""
        self.episodes_ran = 0
        self.steps_ran = 0
        self.steps_this_episode = 0 
        self.steps_moving_towards = 0
        
        """Initialize variables needed for graphing effeciency of agent"""
        self.progress_list = []
        self.progress_steps = 0
        
        """Initialize variables for rendering"""
        self.window = pygame.display.set_mode((self.WIN_WIDTH, self.WIN_HEIGHT))
        pygame.init()
        pygame.font.init()
        
        """Initialize variables needed for graphing"""
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'o')
        self.line_best_fit, = self.ax.plot([], [], linestyle='--', color='r', label='Line of best fit')
        self.ax.set_title("Figure 1: % total steps heading towards prey each episode")
        self.ax.set_xlabel('Episodes ran')
        self.ax.set_ylabel("% Total steps that were heading towards prey")
        self.ax.legend()
        self.ax.grid(True)
        self.ax.set_ylim(0, 100)
        
    def reset(self, seed=None, options=None):
        """Update graph detailing effeciency increase over time of agent"""
        if self.steps_this_episode > 0:
            self.progress_list.append(round(self.progress_steps / self.steps_this_episode * 100, 2))
            self.progress_steps = 0
            x_list = np.array(list(range(1, self.episodes_ran + 1)))
            y_list = np.array(copy.deepcopy(self.progress_list))
            
            if len(x_list) > 5 and len(y_list) > 5:
                self.line.set_data(x_list, y_list)
                
                """Determine line of best fit"""
                a, b = np.polyfit(x_list, y_list, 1)
                self.line_best_fit.set_data(x_list, a*x_list+b)
                
                self.ax.relim()
                self.ax.autoscale_view()
                plt.savefig('progress_plot.png', dpi=300)
        
        """Process variables counting steps and episodes"""
        self.episodes_ran += 1
        self.steps_this_episode = 0
        self.steps_moving_towards = 0

        """Reset variables needed for predator and prey movement, spawn both a new random position"""
        self.wolves = []
        self.predator_poses = []
        self.prev_predator_poses = []
        for i in range(0, 1):
            wolfX, wolfY = (random.randrange(0, 1000), random.randrange(0, 750))
            self.wolves.append(Predator(wolfX, wolfY))
            self.predator_poses.append((wolfX + 10, wolfY + 10))
            self.prev_predator_poses.append((wolfX + 10, wolfY + 10))
            
        preyX, preyY = (random.randrange(0, 1000), random.randrange(0, 750))
        self.prey = Prey(preyX, preyY)
        self.prey_pos = (preyX + 15, preyY + 15)
        self.prev_prey_pos = (preyX + 15, preyY + 15)
        
        """Calculate initial distance between predator and prey"""
        pred1_d = self.distance(self.predator_poses[0], self.prey_pos)
        """Point the predator in the direction of prey / set to random pos"""
        desired_angle = math.degrees(math.atan2(self.prev_predator_poses[0][1] - self.prey_pos[1], self.prev_predator_poses[0][0] - self.prey_pos[0]))
        self.current_pred_angle = (- (desired_angle + 90 + random.randrange(-20, 20)))%360 
        
        """Initial deviation between perfect angle and actual angle"""
        desired_heading = math.degrees(math.atan2(self.predator_poses[0][1] - self.prey_pos[1], self.predator_poses[0][0] - self.prey_pos[0]))
        current_heading = self.wolves[0].get_current_heading()
        prey_heading = self.prey.get_current_heading()
        
        """Point prey to opposite direction of predator / set to run in a circle"""
        self.current_prey_angle = (- (desired_angle + 90))%360
        
        observation = np.array([pred1_d, current_heading, desired_heading, prey_heading])
        return observation, {}
        
    def step(self, action): 
        self.steps_ran += 1
        self.steps_this_episode += 1
        
        """Move prey"""
        self.prev_prey_pos = copy.deepcopy(self.prey_pos)
        self.current_prey_angle = (self.current_prey_angle + 0.5)%360
        self.prey.rotation = self.current_prey_angle
        
        self.prey.move(math.radians(self.current_prey_angle))
        self.prey_pos = (self.prey.originx, self.prey.originy)
        
    
        """Move predator based on action"""
        self.prev_predator_poses = copy.deepcopy(self.predator_poses)
        self.current_pred_angle = (self.current_pred_angle + action[0]*5) % 360
        wolf = self.wolves[0]
        wolf.rotation = self.current_pred_angle
        wolf.move(self.current_pred_angle)
        self.predator_poses[0] = (wolf.originx, wolf.originy)
        
        """Reward system"""
        terminated = False
        reward = 0           
        if wolf.collides_with_prey(self.prey.originx, self.prey.originy):
            terminated = True
            reward += 50
        
        """Check if predator is closing in on prey"""
        pred1_d = self.distance(self.predator_poses[0], self.prey_pos)
        prev_pred1_d = self.distance(self.prev_predator_poses[0], self.prev_prey_pos)
        pred1_closing = prev_pred1_d - pred1_d >= 0.5
            
        """Calculate deviation between past angle and current angle"""
        desired_heading = math.degrees(math.atan2(self.predator_poses[0][1] - self.prey_pos[1], self.predator_poses[0][0] - self.prey_pos[0]))
        current_heading = wolf.get_current_heading()
        prey_heading = self.prey.get_current_heading()
        """Check if the predator has been in this position (estimated), essentialy check for circling"""
        """Reward for small deviation and closing in, further reward for moving in 
        straight line (direct movement as opposed to ineffecient circling), punishment
        for heading in opposite direction to the prey"""

        within_FoV, _ = wolf.line_collision(self.prey)
        if within_FoV:
            self.progress_steps += 1
            closing = 0
            if pred1_closing:
                closing = 2
            reward += 1 + 0.001 * (1000-pred1_d) + closing
        
        """Render every 20 episodes, and always first 10 episodes for interaction loops"""
        if self.episodes_ran <= 5 or (self.episodes_ran%20==0):
            self.render(reward)
            
        info, truncated = {}, False 
        observation = np.array([pred1_d, current_heading, desired_heading, prey_heading])
        return observation, reward, terminated, truncated, info

    def render(self, reward):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                quit()
                
        rect_surface = pygame.Surface((self.WIN_WIDTH, self.WIN_HEIGHT)) #rectangle surface
        pygame.draw.rect(rect_surface, (100, 150, 100), (0, 0, self.WIN_WIDTH, self.WIN_HEIGHT))
        self.window.blit(rect_surface, (0, 0))
            
        font = pygame.font.Font(None, 36)
        text = f'Episode: {self.episodes_ran}   Step: {self.steps_ran}  Reward: {reward}'
        text_surface = font.render(text, True, (255, 255, 255))
        self.window.blit(text_surface, (50, 50))
            
        for wolf in self.wolves:
            wolf.draw(self.window, self.prey)
        self.prey.draw(self.window)
        
        frame = pygame.surfarray.array3d(self.window)
        frame = np.transpose(frame, (1, 0, 2))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        pygame.display.update()
       
    def close(self):
        pygame.quit()
        sys.exit()
    
    def distance(self, coord1, coord2):
        (x1, y1), (x2, y2) = coord1, coord2
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
if __name__ == "__main__":
    """Register environment"""
    register(
        id='SimWolf-v0',  #unique ID for your environment
        entry_point='singleEnv:SimWolf',  #module and class name
        max_episode_steps=1024,  #maximum number of steps per episode, shorter episodes can reduce chance of being stuck in a local optimum
    )
    """Create vectorized environment"""
    vec_env = make_vec_env("SimWolf-v0", n_envs=1)
    
    type = input("Start training or interaction loop? (1 | 2): ")
    if type != '2':
        if type=="1":
            """Smaller n_steps could prevent agent opting for frequent rewards. Added deeper neural network. Lower learning rate
            avoids overshooting optimal solution"""
            policy_kwargs = dict(net_arch=[256, 128, 64]) #define 4 hidden layers as opposed to default 2, detect patterns
            model = PPO("MlpPolicy", vec_env, gamma=0.999, ent_coef=0.012, learning_rate=0.0007, n_steps=1024, normalize_advantage=True, verbose=1) 

        model.learn(total_timesteps=855_000, progress_bar=True)
        model.save("model")
        plt.savefig('progress_plot_backup.png', dpi=300)
        del model
        
    """Testing the model, running the finallized hunting strategy"""
    model = PPO.load("single_moving")
    vec_env = make_vec_env("SimWolf-v0", n_envs=1)
    obs = vec_env.reset()

    show = input("Start interaction loop? (Y/N): ")
    if show.lower() == 'y':
        for i in range(10000):
            action, _state = model.predict(obs, deterministic=True) 
            obs, reward, terminated, info = vec_env.step(action)
    vec_env.close()