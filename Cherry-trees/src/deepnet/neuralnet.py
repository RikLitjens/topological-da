import torch
import torch.nn as nn
import random as rd
import numpy as np


class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Flatten(start_dim=1),
            nn.Linear(1 * 2 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

        # I'd do this differently though..
        self.final_fc = nn.Linear(6, 1)       # Final fully connected layer leading to <is_valid> (2d) and <category> (5d) one-hot encoded vector

    # def forward(self, position_elevation: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # position_elevation shape: torch.Tensor([Batch, Length]) --> [x, 4]
        # image shape: torch.Tensor([Batch, Channel, Height, Width]) --> [x, 1, 32, 16]

        # Process image and concatenate
        img_vector = self.layers(image)

        # Final linear layer
        data = self.final_fc(img_vector)
        output = torch.sigmoid(data)
        return output.squeeze(1)