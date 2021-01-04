import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torch import nn
from torch import optim
from model.DCGAN_architecture import Generator, Discriminator
import matplotlib.pyplot as plt
import numpy as np
from noise import Noise
from torchvision.utils import save_image

def plot_training_image(dataloader, device):
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Image")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()

saved_model_path = "saved_model/"
path_result = "result/"
dataroot = 'dataset/'
image_size = 28
batch_size = 64
workers = 2
nz = 25
n_epochs = 15
beta1 = 0.5

dataset= datasets.MNIST(root= dataroot, train= True, transform= transforms.Compose([
    transforms.Resize(size=image_size),
    transforms.CenterCrop(size=image_size),
    transforms.ToTensor()]))

# dataset = datasets.MNIST(root=dataroot, transform=transforms.Compose([
#     transforms.Resize(size=image_size),
#     transforms.CenterCrop(size=image_size),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# ]))

dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

netG = Generator(n_feature_maps=16, n_latents=nz, n_channels=1)
netD = Discriminator(n_channels=1, n_feature_maps=16)
netD.to(device=device)
netG.to(device=device)

# BCE loss function
criterion = nn.BCELoss()

# batch of noise
fixed_noise = torch.randn(batch_size, nz, 1, 1, device= device)

# label
real_label = 1
fake_label = 0

# optimizer
optimizerD = optim.Adam(netD.parameters(), lr=0.0015, betas= (beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0015, betas= (beta1, 0.999))



# train
img_list = []
G_loss = []
D_loss = []
iters = 0
# plot_training_image(dataloader= dataloader, device= device)
# netG.apply(nn.init.normal)
for epoch in range(n_epochs):
    for i, data in enumerate(dataloader, 0):

        # train D with real images
        optimizerD.zero_grad()
        real_img = data[0].to(device)
        label = torch.full(size=(real_img.size(0),), fill_value=real_label, dtype=torch.float, device=device)
        output = netD(real_img).view(-1)
        Dloss_real = criterion(output, label)
        Dloss_real.backward()

        #train D with fake images
        noise = Noise.normal_noise(batch_size=real_img.size(0), dim=nz, std=1.)
        fake_img = netG(noise)
        label.fill_(fake_label)
        output = netD(fake_img).view(-1)
        Dloss_fake = criterion(output, label)
        Dloss_fake.backward()
        optimizerD.step()

        Dloss = (Dloss_fake + Dloss_real) / 2


        # train G
        optimizerG.zero_grad()
        noise = Noise.normal_noise(batch_size=real_img.size(0), dim=nz, std=1.)
        fake_img = netG(noise)
        label.fill_(real_label)
        output = netD(fake_img).view(-1)
        Gloss = criterion(output, label)

        Gloss.backward()
        optimizerG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, n_epochs, i, len(dataloader),
                     Dloss.item(), Gloss.item()))
        iters += 1
        G_loss.append(Gloss.item())
        D_loss.append(Dloss.item())
        if iters % 500 == 0:
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img = vutils.make_grid(fake, padding=2, normalize=True)
            save_image(tensor= img, fp= path_result + str(iters) + ".jpg", normalize= True)



torch.save(netG.state_dict(), saved_model_path +
           "G_state_dict: n_epoch= {n_epoch}, z_dim= {dim}.pt".format(n_epoch= n_epochs, dim= nz))
torch.save(netD.state_dict(), saved_model_path +
           "D_state_dict: n_epoch= {n_epoch}, z_dim= {dim}.pt".format(n_epoch= n_epochs, dim= nz))

plt.figure(figsize= (10, 5))
plt.title("Generator and Discriminator loss")
plt.plot(G_loss, label= "G loss")
plt.plot(D_loss, label= "D loss")
plt.xlabel("iterations")
plt.ylabel("loss")
plt.show()