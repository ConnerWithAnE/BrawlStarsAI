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

import controller
import environment
import imageController
import net

MAX_MEMORY = 1000
LR = 0.001
BATCH_SIZE = 10

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 10 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.threshold = 0.5
        
        
        self.model = net.ConvNet()

        self.trainer = net.QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_action(self, state):
        self.epsilon = 10 - self.n_games
        actions = [0,0,0,0,0,0]
        if random.randint(0,200) < self.epsilon:
            actions = [[random.randint(0,1) for x in range(0,6)]]
        else:
            pred = self.model(state)
            actions = (pred > self.threshold).float().tolist()
        return actions

    def remember(self, state, action, reward, next_state, alive):
        self.memory.append((state, action, reward, next_state, alive))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, alives = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, alives)

    def train_short_memory(self, state, action, reward, next_state, alive):
        self.trainer.train_step(state, action, reward, next_state, alive)


def train():
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0
        agent = Agent()
        im_controller = imageController.ImageController()
        control = controller.Controller()
        ahk = control.getAHK()
        win = control.getWIN()
        env = environment.Environment(im_controller)
        state = env.reset(control)
        
        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            state_tensor = state_tensor.unsqueeze(0)

            actions = agent.get_action(state_tensor)
            
            #binary_actions = (q_values > threshold).float().tolist()

            #binary_qs = (q_values > threshold).float()
            
            print(f"Actions {actions}")

            next_state, reward, alive = env.step(actions, ahk)
            next_state = torch.tensor(state, dtype=torch.float32)
            next_state = state_tensor.unsqueeze(0)

            print(f"Reward: {reward}")

            agent.train_short_memory(state_tensor, actions, reward, next_state, alive)

            agent.remember(state_tensor, actions, reward, next_state, alive)

            print(f"Is alive: {alive}")
            if not alive:
                agent.n_games+=1
                print(f"Game Num {agent.n_games}:\n    Teams: {env.teams}\n    Cubes: {env.cubes}")
                control.exitGame()
                time.sleep(5)
                agent.train_long_memory()
                state = env.reset(control)
            
                # Save the entire model
                if agent.n_games % 100 == 0:
                    torch.save(agent.model, './trained_model.pth')   

'''
def train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    im_controller = imageController.ImageController()
    control = controller.Controller()

    ahk = control.getAHK()
    win = control.getWIN()
    env = environment.Environment(im_controller)

    learning_rate = 0.001

    num_epochs = 10
    gamma = 0 # discount rate
    epsilon = 0 # Randomness


    cnn_model = net.ConvNet()

    threshold = 0.5
    

    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=learning_rate) 
    loss_fn = torch.nn.CrossEntropyLoss() 

    for epoch in range(num_epochs):
        time.sleep(5)
        state = env.reset(control)

        while True:
            #print(state)
            #print("hit")
            state_tensor = torch.tensor(state, dtype=torch.float32)
        
            state_tensor = state_tensor.unsqueeze(0)
            #state_tensor = state_tensor.to(device)
           
            q_values = cnn_model(state_tensor)
            
            binary_actions = (q_values > threshold).float().tolist()

            binary_qs = (q_values > threshold).float()

            print(f"Actions {binary_actions}")

            next_state, reward, alive = env.step(binary_actions, ahk)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)#.permute(0, 3, 1, 2)
            
            print(next_state.shape)

            td_target = reward + gamma * q_values.max(dim=1)[0] * (1.0)
            current_q_value = binary_qs.gather(1, torch.arange(q_values.size(1)).unsqueeze(0))
            td_target_expanded = td_target.unsqueeze(1).expand_as(current_q_value)

           
            # Compute the loss and perform a gradient descent step
            loss = loss_fn(current_q_value, td_target_expanded)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            print(f"Is alive: {alive}")
            if not alive:
                print(f"Epoch {epoch}:\n    Teams: {env.teams}\n    Cubes: {env.cubes}")
                control.exitGame()
                break
        print("Over")
        
    
    # Save the entire model
    torch.save(cnn_model, 'trained_model.pth')

    # OR save only the model parameters (state_dict)
    torch.save(cnn_model.state_dict(), 'trained_model_state_dict.pth')
'''
if __name__ == '__main__':
    train()