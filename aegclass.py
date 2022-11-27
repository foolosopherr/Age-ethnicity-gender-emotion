import torch.nn as nn


class MultilabelCNN(nn.Module):
    def __init__(self, target_features):
        super().__init__()
        
        self.CNNModel = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), #24
            # nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), #12
            # nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), #6
            nn.LeakyReLU(),
            # # nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), #3
            # nn.Dropout(0.1),
            nn.AdaptiveAvgPool2d((1,1)), #flatten
            nn.Flatten()
            # nn.LeakyReLU()
        )

        self.DNNModel = nn.Sequential(
            nn.Linear(256, 128),
            # # nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            # nn.Dropout(0.1),
            nn.LeakyReLU(),
            # nn.Dropout(0.4), 
            nn.Linear(64, 32),
            # nn.Dropout(0.1), 
            nn.LeakyReLU())
        
        if target_features > 20:
          self.target_layer = nn.Linear(32, 1)
        else:
          self.target_layer = nn.Linear(32, target_features)
        
    def forward(self, x):
        output = self.CNNModel(x).squeeze()
        output = self.DNNModel(output)

        pred = self.target_layer(output)
        
        return pred