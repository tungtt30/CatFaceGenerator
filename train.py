
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from model import Generator, Discriminator
import matplotlib.pyplot as plt
import numpy as np


train_path = "./data/train"
test_path = "./data/test"
val_path = "./data/val"


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
val_dataset = datasets.ImageFolder(root=val_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)



nz = 100            #latent vector
ngf = 128            # Feature maps G
ndf = 128            # Feature maps D
nc = 3              # Channel
lr = 0.0002         
beta1 = 0.5         
num_epochs = 50     
batch_size = 64     
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator(nz, ngf, nc).to(device)
discriminator = Discriminator(nc, ndf).to(device)


#Test output shape
noise = torch.randn(16, nz, 1, 1).to(device)  
fake_images = generator(noise)
print(fake_images.size())  

real_images = torch.randn(16, 3, 128, 128).to(device) 
output = discriminator(real_images)
print(output.size()) 


criterion = nn.BCELoss()
optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
# generator.apply(weights_init)
# discriminator.apply(weights_init)



def plot_images(images):
    images = (images + 1) / 2.0
    grid = np.transpose(torchvision.utils.make_grid(images, nrow=8, padding=2, normalize=True).cpu(), (1, 2, 0))
    plt.figure(figsize=(8, 8))
    plt.imshow(grid)
    plt.axis("off")
    plt.show()


for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0):
        generator.train()
        discriminator.train()

        #train d
        real_images, _ = data
        real_images = real_images.to(device)
        b_size = real_images.size(0)
        real_labels = torch.ones(b_size, device=device)

        
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_images = generator(noise)
        fake_labels = torch.zeros(b_size, device=device)

        
        optimizerD.zero_grad()
        output_real = discriminator(real_images).view(-1)
        loss_real = criterion(output_real, real_labels)
        output_fake = discriminator(fake_images.detach()).view(-1)
        loss_fake = criterion(output_fake, fake_labels)
        lossD = loss_real + loss_fake
        lossD.backward()
        optimizerD.step()

        #train g
        optimizerG.zero_grad()
        output = discriminator(fake_images).view(-1)
        lossG = criterion(output, real_labels) 
        lossG.backward()
        optimizerG.step()

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch {i}/{len(train_loader)} "
                  f"Loss D: {lossD.item():.8f}, Loss G: {lossG.item():.8f}")

    if epoch == 49:
        with torch.no_grad():
            fixed_noise = torch.randn(64, nz, 1, 1, device=device)
            fake_images = generator(fixed_noise)
            plot_images(fake_images)

torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")