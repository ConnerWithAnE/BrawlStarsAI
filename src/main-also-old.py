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


import controller
import environment
import imageController
import net

MAX_MEMORY = 1000
BATCH_SIZE = 10

n_games = 0
epsilon = 0 # randomness
gamma = 0 # discount rate
memory = deque(maxlen=MAX_MEMORY)
threshold = 0.5
learning_rate = 0.001



def train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    im_controller = imageController.ImageController()
    control = controller.Controller()

    ahk = control.getAHK()
    win = control.getWIN()
    env = environment.Environment(im_controller)

    model = net.ConvNet(0.003)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
    loss_fn = nn.MSELoss()

    state = env.reset(control)

    '''
    for epoch in range(num_epochs):
        time.sleep(5)
        state = env.reset(control)
    '''
    while True:
        #print(state)
        #print("hit")
        print(np.reshape(state, (1,516,918)).shape)
        state_tensor = torch.tensor(np.reshape(state, (1,516,918)), dtype=torch.float32)
        #state_tensor = state_tensor.unsqueeze(0)
        print(state_tensor.shape)
        #state_tensor = state_tensor.to(device)
        
        q_values = model(state_tensor)
        
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
            n_games+=1
            print(f"Game Num {n_games}:\n    Teams: {env.teams}\n    Cubes: {env.cubes}")
            control.exitGame()
            time.sleep(5)
            state = env.reset(control)
        
             # Save the entire model
            if n_games % 100 == 0:
                torch.save(model, './trained_model.pth')

             # OR save only the model parameters (state_dict)
            #torch.save(cnn_model.state_dict(), 'trained_model_state_dict.pth')

if __name__ == '__main__':
    train()