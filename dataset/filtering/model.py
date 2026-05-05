import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, image_size, num_channels, ngpu):
        super(Classifier, self).__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(num_channels, 64, 4, 2, 1), # 32
            nn.MaxPool2d(2), # 16
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), # 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 32, 4, 2, 1),  # 4
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Flatten(), # 4*4*32
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, discriminator_input):
        return self.main(discriminator_input)