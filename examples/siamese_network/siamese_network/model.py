import math
import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self, n_mfcc, sf, cut_length, hop_length, n_components):
        super(SiameseNetwork, self).__init__()
        input_t = math.ceil((sf * cut_length / hop_length)) + 1
        output_c = math.floor((n_mfcc - 7) / 2 + 1) - 10
        output_t = math.floor((input_t - 7) / 2 + 1) - 10
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3), nn.ReLU(),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(256 * output_c * output_t, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_components),
        )

    def forward(self, x1, x2):
        y1, y2 = self.conv_layers(x1), self.conv_layers(x2)
        return self.linear_layers(y1.view(y1.size(0), -1)), \
            self.linear_layers(y2.view(y1.size(0), -1))
