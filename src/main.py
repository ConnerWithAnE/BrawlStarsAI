import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torch.optim as optim
from torchvision import transforms
import numpy as np
import datetime
import sys


import controller
import environment
import imageController
import net



def train():

    im_controller = imageController.ImageController()
    control = controller.Controller()

    ahk = control.getAHK()
    win = control.getWIN()
    env = environment.Environment(im_controller)

    cnn_model = net.SimpleCNN(1, 516,918,6)
    num_epochs = 10
    gamma = 0.99
    epsilon = 0.1
    threshold = 0.5

    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCELoss() 

    for epoch in range(num_epochs):
        state = env.reset(control)

        while True:
            print(state)
            state_tensor = torch.tensor(state, dtype=torch.float32)
            print(state_tensor.shape)
            state_tensor = state_tensor.unsqueeze(0)#.unsqueeze(0)#.permute(0, 3, 1, 2)
            #state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
            q_values = cnn_model(state_tensor)
            
            binary_actions = (q_values > threshold).float().tolist()

            print(f"Actions {binary_actions}")

            next_state, reward, alive = env.step(binary_actions, ahk)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)#.permute(0, 3, 1, 2)
            with torch.no_grad():
                
                next_q_values = cnn_model(next_state_tensor)
                next_q_values.shape
                td_target = reward + gamma * next_q_values.max(dim=1)[0] * (1.0)

            current_q_value = q_values.gather(1, torch.arange(q_values.size(1)).unsqueeze(0))

            td_target_expanded = td_target.unsqueeze(1).expand_as(next_q_values)

            print(current_q_value.shape)
            print(td_target.shape)
            # Compute the loss and perform a gradient descent step
            loss = loss_fn(current_q_value, td_target_expanded)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

            if not alive:
                control.exitGame()
                break
        
    
    # Save the entire model
    torch.save(cnn_model, 'trained_model.pth')

    # OR save only the model parameters (state_dict)
    torch.save(cnn_model.state_dict(), 'trained_model_state_dict.pth')


train()