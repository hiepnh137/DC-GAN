import torch
import torch.nn as nn
from model.DCGAN_architecture import Generator
from noise import Noise
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

nz= 25
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
saved_model_path = "saved_model/"
netG = Generator(n_feature_maps=16, n_latents=nz, n_channels=1).to(device= device)
state_dict = torch.load( saved_model_path +
           "G_state_dict: n_epoch= {n_epoch}, z_dim= {dim}.pt".format(n_epoch= 10, dim= nz))
netG.load_state_dict(state_dict)

z = Noise.normal_noise(batch_size= 64, dim= nz, std= 1.)
fake = netG(z).detach().cpu()
img = vutils.make_grid(fake, padding= 2, normalize= True)
plt.imshow(np.transpose(img, (1, 2, 0)))

plt.show()