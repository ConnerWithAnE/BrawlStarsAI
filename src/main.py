import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torch.optim as optim
from torchvision import transforms
import numpy as np
import time
import datetime
import sys
from collections import deque
import random
import matplotlib.pyplot as plt
import subprocess
import shutil
import os
import joblib

import controller
import environment
import imageController
import net

class Agent:

    def __init__(self, gamma, epsilon, lr, batch_size, max_mem=1000, eps_end=0.01, eps_dec=5e-4):
        self.epsilon = epsilon # randomness
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.gamma = gamma # discount rate
        self.mem_size = max_mem #deque(maxlen=MAX_MEMORY)
        self.threshold = 0.05
        self.mem_ctr = 0
        self.lr = lr
        self.action_space = [0,0,0,0,0,0]
        self.batch_size = batch_size
        self.num_rand = 0
        self.num_chose = 0
        self.game = 0
        
        self.Q_eval = net.ConvNet(self.lr)
        self.optimizer = optim.Adam(self.Q_eval.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.state_memory = np.zeros((self.mem_size, 1, 516, 918), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, 1, 516, 918), dtype=np.float32)

        self.action_memory = np.zeros((self.mem_size, 1, 6), dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)


        
    def store_transition(self, state, action, reward, state_, done) :
        index = self.mem_ctr % self.mem_size
        self.state_memory[index] = state
        #print(f"state_mem in transition {self.state_memory[index].shape}")
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_ctr += 1
    
    def choose_action(self, observation):
        if  np.random.random() > self.epsilon:
            state = torch.tensor(np.array(observation, dtype=np.float32)).to(self.Q_eval.device)
            #print(f"unsqueezed state: {state.dtype}")
            actions = self.Q_eval.forward(state)
            print(actions)
            actions = (actions > self.threshold).int().tolist()
            print(f"Chosen Actions: {actions}")
            self.num_chose += 1
        else:
            actions = [[random.randint(0,1) for x in range(0,6)]]
            print(f"Random Actions: {actions}")
            self.num_rand += 1
        return actions
    

    def learn(self):
        if self.mem_ctr < self.batch_size:
            return
        
        self.optimizer.zero_grad()

        max_mem = min(self.mem_ctr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        #print(f"state_memory: {self.state_memory.shape}")
        #print(f"state_memory[batch] {self.state_memory[batch].dtype}")
       # print(f"batch num: {batch}")
       # print(f"state_memory_batch: {self.state_memory[batch].shape}")

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        #print(f"state_batch: {state_batch.shape}")
        #print(f"state_batch_indexed {state_batch[batch_index, action_batch]}")

        q_eval = self.Q_eval.forward(state_batch)
       # print(q_eval)
        #print(q_eval.shape)
        q_eval = torch.sum(q_eval, dim=1)
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        #print(q_next)
        #print(q_next.shape)
        #print(reward_batch)
        #print(reward_batch.shape)

        #print(torch.max(q_next, dim=1)[0])

        q_target = reward_batch + self.gamma * torch.sum(q_next, dim=1)

       # print(q_target.shape)
        ##print(q_target)
        #print(q_eval)
        #print(q_eval.squeeze().shape)

        loss = self.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

          


def train():
        plot_cubes = []
        plot_eps_history = []
        plot_reward_history = []
        plot_teams_remaining = []

        #fig = plt.figure()

        n_games = 500

        total_score = 0
        record = 0
        agent = Agent(gamma=0.99, epsilon=1.0, batch_size=16, eps_end=0.01, lr=0.003)
        
        if os.path.exists("./saves/models/trained_model_1_games.pth"):
            print('hit')
            agent = joblib.load("./saves/models/trained_model_1_games.pth")
           
        im_controller = imageController.ImageController()
        control = controller.Controller()
        ahk = control.getAHK()
        win = control.getWIN()
        env = environment.Environment(im_controller)
        #state = env.reset(control)

       
    
        print(agent.epsilon)

        while agent.game < n_games:
            
            checkRunning(control)
            state = env.reset(control)
            alive = True
            total_reward = 0
            exit_type = 0
            agent.num_rand = 0
            agent.num_chose = 0
            ahk.key_up('w')
            ahk.key_up('a')
            ahk.key_up('s')
            ahk.key_up('d')
            win.activate()

            while alive:

                if ahk.key_state('h') == 1:
                    alive = False

                #print(f"start state {state.shape}")
                #print(f"start state unsqueezed: {state.unsqueeze(0).shape}")

                actions = agent.choose_action(state)
            
                #print(f"Actions {actions}")

                state_, reward, alive, exit_type = env.step(actions, ahk)
                total_reward += reward
                agent.store_transition(state, actions, reward, state_, alive)
                agent.learn()
                state = state_

                if checkRunning(control) == False:
                    alive = False
                    exit_type = 2
                
                print(f"Is alive: {alive}\n")
                
            print(f"Game Num {agent.game+1}:\n   Teams: {env.teams}\n   Cubes: {env.cubes}\n   Random Moves: {agent.num_rand}\n   Chosen Moves: {agent.num_chose}\n   Epsilon: {agent.epsilon}")
            plot_cubes.append(env.cubes)
            plot_eps_history.append(agent.epsilon)
            plot_teams_remaining.append(env.teams)
            plot_reward_history.append(total_reward)
            time.sleep(1)
            if exit_type == 1 or im_controller.winScreen():
                time.sleep(1)
                control.exitGameWin()
            elif exit_type == 0:
                control.exitGameLose()
            checkRunning(control)
            time.sleep(5)
                    #agent.train_long_memory()
            #state = env.reset(control)
                
                    # Save the entire model
            if agent.game % 2 == 0:
                
                joblib.dump(agent, f"./saves/models/trained_model_{agent.game+1}_games.pth")
                print("Save Complete")
                x = [j for j in range(agent.game+1)]
                plt.figure()
                #plt.plot(x, plot_eps_history, label="Epsilon")
                plt.xlabel("Epsilon")
                plt.plot(plot_eps_history, plot_reward_history, label="Rewards")
                plt.plot(plot_eps_history, plot_cubes, label="Power Cubes")
                plt.plot(plot_eps_history, plot_teams_remaining, label="Teams Remaining")
                plt.legend()
                plt.savefig(f"./saves/plots/training_plot_{agent.game+1}_games.png")
            agent.game += 1


def checkRunning(control):
    running = subprocess.run(['C:\\Users\\conne\\Desktop\\BrawlStarsAI\\src\\platform-tools\\adb.exe', 'shell', 'pidof', 'com.supercell.brawlstars'],check=True, capture_output=True, text=True)
    if running == None:
        print("Restarting Game")
        control.restartGame()
        time.sleep(5)
        return False
    return True


if __name__ == '__main__':
    train()