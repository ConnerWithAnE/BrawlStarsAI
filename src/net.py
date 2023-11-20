import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvNet(nn.Module):
    def __init__(self, lr):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=2)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5)

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=2)
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5)
        

        self.fc1 = nn.Linear(16*21*47, 120)
        self.relu = nn.ReLU()  # Adjust the input size based on your image size
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 6)
        self.flat = nn.Flatten()

        self.n_actions = 6
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    

    def forward(self, x):
        #print(f"input shape {x.shape}")
        out = self.conv1(x)
        out = self.relu(out)
        #out = self.pool(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.relu(out)
        out = self.conv6(out)
        out = self.relu(out)
        out = self.conv7(out)
        out = self.relu(out)
        out = self.conv8(out)
        out = self.relu(out)
        #out = self.pool(out)
        
        #out = self.conv3(out)
        #out = self.pool2(out)
        #out = self.conv4(out)
        print(f"conv shape {out.shape}")
        out = out.view(-1, 16*21*47)

        print(f"conv flat shape {out.shape}")

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)

        #print(out.shape)

        return out
