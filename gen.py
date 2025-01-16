import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch
from model import Generator


gen_model = Generator(100, 128, 3)
state = torch.load('generator.pth', weights_only=True)
gen_model.load_state_dict(state)



def plot_images(images):
    images = (images + 1) / 2.0
    grid = np.transpose(torchvision.utils.make_grid(images, nrow=8, padding=2, normalize=True).cpu(), (1, 2, 0))
    plt.figure(figsize=(10, 10))
    plt.imshow(grid)
    plt.axis("off")
    plt.show()

with torch.no_grad():
    fixed_noise = torch.randn(64, 100, 1, 1)
    fake_images = gen_model(fixed_noise)
    plot_images(fake_images)