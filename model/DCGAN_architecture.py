import torch
import torch.nn as nn
from torchsummary import summary


class Generator(nn.Module):
    def __init__(self, n_feature_maps, n_latents, n_channels):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels= n_latents, out_channels= n_feature_maps*4, kernel_size= 4, stride= 1, padding= 0),
            nn.BatchNorm2d(num_features= n_feature_maps*4),
            nn.ReLU(inplace= True),

            nn.ConvTranspose2d(in_channels= n_feature_maps*4, out_channels= n_feature_maps*2, kernel_size= 4, stride= 2, padding= 1),
            nn.BatchNorm2d(num_features= n_feature_maps*2),
            nn.ReLU(inplace= True),

            nn.ConvTranspose2d(in_channels= n_feature_maps*2, out_channels= n_feature_maps*1, kernel_size= 4, stride= 2, padding= 1),
            nn.BatchNorm2d(num_features= n_feature_maps*1),
            nn.ReLU(inplace= True),

            # nn.ConvTranspose2d(in_channels= n_feature_maps*2, out_channels= n_feature_maps, kernel_size= 4, stride= 2, padding= 3),
            # nn.BatchNorm2d(num_features= n_feature_maps),
            # nn.ReLU(inplace= True),
            #
            nn.ConvTranspose2d(in_channels= n_feature_maps, out_channels= n_channels, kernel_size= 4, stride= 2, padding= 3),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, n_channels, n_feature_maps):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels= n_channels, out_channels= n_feature_maps, kernel_size= 4, stride= 2, padding= 3),
            nn.LeakyReLU(negative_slope= 0.02, inplace= True),

            nn.Conv2d(in_channels= n_feature_maps, out_channels= n_feature_maps*2, kernel_size= 4, stride= 2, padding= 1),
            nn.BatchNorm2d(num_features= n_feature_maps*2),
            nn.LeakyReLU(negative_slope= 0.02, inplace= True),

            nn.Conv2d(in_channels= n_feature_maps*2, out_channels= n_feature_maps*4, kernel_size= 4, stride= 2, padding= 1),
            nn.BatchNorm2d(num_features= n_feature_maps*4),
            nn.LeakyReLU(negative_slope= 0.02, inplace= True),
            #
            # nn.Conv2d(in_channels= n_feature_maps*4, out_channels= n_feature_maps*8, kernel_size= 4, stride= 2, padding= 1),
            # nn.BatchNorm2d(num_features= n_feature_maps*8),
            # nn.LeakyReLU(negative_slope= 0.02, inplace= True),

            nn.Conv2d(in_channels= n_feature_maps*4, out_channels= 1, kernel_size= 4, stride= 1, padding= 0),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

