import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):

        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Block 1: Input [nc, 128, 128] -> [ndf, 64, 64]
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: [ndf, 64, 64] -> [ndf * 2, 32, 32]
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3: [ndf * 2, 32, 32] -> [ndf * 4, 16, 16]
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 4: [ndf * 4, 16, 16] -> [ndf * 8, 8, 8]
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output Block: [ndf * 8, 8, 8] -> [1, 1, 1]
            nn.Conv2d(ndf * 8, 1, 8, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1)


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):

        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Block 1: Input (nz) -> [ngf * 8, 4, 4]
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # Block 2: [ngf * 8, 4, 4] -> [ngf * 4, 8, 8]
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # Block 3: [ngf * 4, 8, 8] -> [ngf * 2, 16, 16]
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # Block 4: [ngf * 2, 16, 16] -> [ngf, 32, 32]
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # Output Block: [ngf, 32, 32] -> [nc, 128, 128]
            nn.ConvTranspose2d(ngf, nc, 4, 4, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)