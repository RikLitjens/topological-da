import torch
import torch.nn as nn
import random as rd
import numpy as np


class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.image_to_vector = nn.Sequential(
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

        # self.linear1 = nn.Linear(10, 128)      # Combined pos/elev and feature vector: 4 + 6 = 1
        self.linear1 = nn.Linear(4, 128)        # only image data
        self.dropout1 = nn.Dropout(p=0.25)      # Dropout layer with 25% dropout
        self.linear2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(p=0.25)

        # I'd do this differently though..
        self.final_fc = nn.Linear(128, 7)       # Final fully connected layer leading to <is_valid> (2d) and <category> (5d) one-hot encoded vector

    def forward(self, position_elevation: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        # position_elevation shape: torch.Tensor([Batch, Length]) --> [x, 4]
        # image shape: torch.Tensor([Batch, Channel, Height, Width]) --> [x, 1, 32, 16]

        # Process image and concatenate
        img_vector = self.image_to_vector(image)

        # with pos encoding
        # data = torch.concat((position_elevation, img_vector), dim=1)

        # without pos encoding
        data = img_vector

        # Linear layer 1
        data = self.dropout1(self.linear1(data))
        data = torch.relu(data)

        # Linear layer 2
        data = self.dropout2(self.linear2(data))
        data = torch.relu(data)

        # Final linear layer
        data = self.final_fc(data)
        output = torch.sigmoid(data)
        return output

def explode_data(histogram, n):
    print("Started new function!")
    histograms = [histogram]
    
    final_histograms = []

    histograms.append(np.flip(histogram, axis=0))
    histograms.append(np.flip(histogram, 1))

    for i in range(len(histograms)):
        current_hist = histograms[i]
        final_histograms.append(current_hist)
        for j in range(n):
            editable = current_hist.copy()
            for k in range(len(editable)):
                for l in range(len(editable[k])):
                    if editable[k][l] > 4:
                        upper_bound = min(255, editable[k][l] * 2)
                        # editable[k][l] = min(rd.randint(1, upper_bound), rd.randint(1, upper_bound)) # Favor the lower values
                        editable[k][l] = rd.randint(5, upper_bound)
            final_histograms.append(editable)
    
    return final_histograms
