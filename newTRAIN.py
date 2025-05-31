import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random


# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DogDataset(Dataset):
    """Custom dataset cho ảnh chó"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        
        # Load tất cả file ảnh
        for file in os.listdir(root_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(root_dir, file))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


def weights_init(m):
    """Initialize network weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """Generator network tối ưu cho ảnh 64x64"""
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # Input: Z latent vector [nz] -> [ngf*8, 4, 4]
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # State size: [ngf*8, 4, 4] -> [ngf*4, 8, 8]
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # State size: [ngf*4, 8, 8] -> [ngf*2, 16, 16]
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # State size: [ngf*2, 16, 16] -> [ngf, 32, 32]
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # State size: [ngf, 32, 32] -> [nc, 64, 64]
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    """Discriminator network tối ưu cho ảnh 64x64"""
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input: [nc, 64, 64] -> [ndf, 32, 32]
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: [ndf, 32, 32] -> [ndf*2, 16, 16]
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: [ndf*2, 16, 16] -> [ndf*4, 8, 8]
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: [ndf*4, 8, 8] -> [ndf*8, 4, 4]
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: [ndf*8, 4, 4] -> [1, 1, 1]
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


class GANTrainer:
    """Class để quản lý quá trình training GAN"""
    
    def __init__(self, data_root, image_size=64, batch_size=64, nz=100, 
                 ngf=64, ndf=64, nc=3, lr=0.0002, beta1=0.5, num_epochs=100):
        
        self.data_root = data_root
        self.image_size = image_size
        self.batch_size = batch_size
        self.nz = nz
        self.ngf = ngf
        self.ndf = ndf
        self.nc = nc
        self.lr = lr
        self.beta1 = beta1
        self.num_epochs = num_epochs
        
        # Setup device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Setup data
        self.setup_data()
        
        # Create networks
        self.netG = Generator(nz, ngf, nc).to(self.device)
        self.netD = Discriminator(nc, ndf).to(self.device)
        
        # Initialize weights
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)
        
        # Setup loss and optimizers
        self.criterion = nn.BCELoss()
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
        
        # Fixed noise for visualization
        self.fixed_noise = torch.randn(64, nz, 1, 1, device=self.device)
        
        # Lists to track progress
        self.G_losses = []
        self.D_losses = []
        
        print("Networks initialized successfully!")
        print(f"Generator parameters: {sum(p.numel() for p in self.netG.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.netD.parameters()):,}")
    
    def setup_data(self):
        """Setup data loading với augmentation mạnh"""
        # Data augmentation để tăng đa dạng data
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])
        
        # Create dataset
        dataset = DogDataset(self.data_root, transform=transform)
        
        # Create dataloader
        self.dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=True, 
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"Dataset loaded: {len(dataset)} images")
        print(f"Batches per epoch: {len(self.dataloader)}")
    
    def train_discriminator(self, real_batch):
        """Train Discriminator"""
        self.netD.zero_grad()
        
        # Train with real images
        real_cpu = real_batch.to(self.device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), 1., dtype=torch.float, device=self.device)
        
        output = self.netD(real_cpu).view(-1)
        errD_real = self.criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()
        
        # Train with fake images
        noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
        fake = self.netG(noise)
        label.fill_(0.)
        
        output = self.netD(fake.detach()).view(-1)
        errD_fake = self.criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        
        errD = errD_real + errD_fake
        self.optimizerD.step()
        
        return errD.item(), D_x, D_G_z1
    
    def train_generator(self, batch_size):
        """Train Generator"""
        self.netG.zero_grad()
        
        label = torch.full((batch_size,), 1., dtype=torch.float, device=self.device)
        noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
        
        fake = self.netG(noise)
        output = self.netD(fake).view(-1)
        
        errG = self.criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        self.optimizerG.step()
        
        return errG.item(), D_G_z2
    
    def save_sample_images(self, epoch, num_samples=64):
        """Save sample images"""
        with torch.no_grad():
            fake = self.netG(self.fixed_noise).detach().cpu()
            
        # Create directory if it doesn't exist
        os.makedirs('generated_samples', exist_ok=True)
        
        # Save grid of images
        vutils.save_image(
            fake,
            f'generated_samples/fake_samples_epoch_{epoch:03d}.png',
            normalize=True,
            nrow=8
        )
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        os.makedirs('checkpoints', exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'netG_state_dict': self.netG.state_dict(),
            'netD_state_dict': self.netD.state_dict(),
            'optimizerG_state_dict': self.optimizerG.state_dict(),
            'optimizerD_state_dict': self.optimizerD.state_dict(),
            'G_losses': self.G_losses,
            'D_losses': self.D_losses,
        }
        
        torch.save(checkpoint, f'checkpoints/gan_checkpoint_epoch_{epoch}.pth')
    
    def train(self):
        """Main training loop"""
        print("Starting Training...")
        
        for epoch in range(self.num_epochs):
            epoch_d_loss = 0
            epoch_g_loss = 0
            
            for i, data in enumerate(self.dataloader, 0):
                # Train Discriminator
                errD, D_x, D_G_z1 = self.train_discriminator(data)
                
                # Train Generator
                errG, D_G_z2 = self.train_generator(data.size(0))
                
                # Save losses
                self.G_losses.append(errG)
                self.D_losses.append(errD)
                
                epoch_d_loss += errD
                epoch_g_loss += errG
                
                # Print statistics
                if i % 50 == 0:
                    print(f'[{epoch}/{self.num_epochs}][{i}/{len(self.dataloader)}] '
                          f'Loss_D: {errD:.4f} Loss_G: {errG:.4f} '
                          f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')
            
            # Print epoch statistics
            avg_d_loss = epoch_d_loss / len(self.dataloader)
            avg_g_loss = epoch_g_loss / len(self.dataloader)
            print(f'Epoch [{epoch+1}/{self.num_epochs}] - Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}')
            
            # Save sample images every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_sample_images(epoch + 1)
            
            # Save checkpoint every 25 epochs
            if (epoch + 1) % 25 == 0:
                self.save_checkpoint(epoch + 1)
        
        print("Training completed!")
        
        # Save final checkpoint
        self.save_checkpoint(self.num_epochs)
        
        # Plot training curves
        self.plot_losses()
    
    def plot_losses(self):
        """Plot training losses"""
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses, label="G")
        plt.plot(self.D_losses, label="D")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('training_losses.png')
        plt.show()
    
    def generate_images(self, num_images=64):
        """Generate new images"""
        self.netG.eval()
        with torch.no_grad():
            noise = torch.randn(num_images, self.nz, 1, 1, device=self.device)
            fake_images = self.netG(noise).detach().cpu()
        
        # Save generated images
        vutils.save_image(
            fake_images,
            'final_generated_dogs.png',
            normalize=True,
            nrow=8
        )
        
        return fake_images


# Cách sử dụng
def main():
    """Main function để chạy training"""
    
    # Set random seed
    set_seed(42)
    
    # Đường dẫn đến folder chứa ảnh chó
    data_root = "path/to/your/dog/images"  # Thay đổi path này
    
    # Tạo trainer với hyperparameters tối ưu cho 20K ảnh
    trainer = GANTrainer(
        data_root=data_root,
        image_size=64,
        batch_size=32,          # Batch size nhỏ hơn cho stability
        nz=100,                 # Noise vector size
        ngf=64,                 # Generator feature maps
        ndf=64,                 # Discriminator feature maps
        nc=3,                   # Number of channels (RGB)
        lr=0.0002,              # Learning rate
        beta1=0.5,              # Beta1 for Adam optimizer
        num_epochs=200          # Số epochs (có thể adjust)
    )
    
    # Bắt đầu training
    trainer.train()
    
    # Generate final images
    print("Generating final sample images...")
    trainer.generate_images(64)


if __name__ == "__main__":
    main()


# === UTILITY FUNCTIONS ===

def load_checkpoint(checkpoint_path, netG, netD, optimizerG, optimizerD):
    """Load checkpoint để continue training"""
    checkpoint = torch.load(checkpoint_path)
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netD.load_state_dict(checkpoint['netD_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    epoch = checkpoint['epoch']
    G_losses = checkpoint['G_losses']
    D_losses = checkpoint['D_losses']
    
    return epoch, G_losses, D_losses


def evaluate_model(generator, num_samples=1000):
    """Evaluate model quality (simple version)"""
    generator.eval()
    
    with torch.no_grad():
        # Generate nhiều ảnh để đánh giá
        all_images = []
        batch_size = 64
        
        for i in range(0, num_samples, batch_size):
            current_batch = min(batch_size, num_samples - i)
            noise = torch.randn(current_batch, 100, 1, 1)
            
            if torch.cuda.is_available():
                noise = noise.cuda()
                
            fake_images = generator(noise).cpu()
            all_images.append(fake_images)
        
        all_images = torch.cat(all_images, dim=0)
        
        # Basic statistics
        print(f"Generated {len(all_images)} images")
        print(f"Image range: [{all_images.min():.3f}, {all_images.max():.3f}]")
        print(f"Image mean: {all_images.mean():.3f}")
        print(f"Image std: {all_images.std():.3f}")
        
    return all_images


# === TRAINING TIPS ===
"""
TIPS CHO TRAINING HIỆU QUẢ VỚI 20K ẢNH CHÓ:

1. DATA PREPARATION:
   - Đảm bảo tất cả ảnh có chất lượng tốt
   - Remove ảnh bị blur, corrupt
   - Ảnh nên có background đa dạng
   - Crop ảnh để tập trung vào con chó

2. HYPERPARAMETER TUNING:
   - Batch size: 32-64 (tùy GPU memory)
   - Learning rate: 0.0002 (có thể giảm xuống 0.0001 nếu unstable)
   - Beta1: 0.5 (quan trọng cho GAN stability)
   - Epochs: 200-500 (monitor loss để quyết định)

3. MONITORING TRAINING:
   - Watch D_x (should be around 0.5)
   - Watch D(G(z)) (should increase over time)
   - Generator loss should decrease gradually
   - Save samples every 10 epochs để check quality

4. COMMON ISSUES:
   - Mode collapse: Giảm learning rate, thêm noise
   - Discriminator too strong: Train G more often
   - Training unstable: Reduce learning rate, add label smoothing

5. IMPROVEMENTS:
   - Thêm Progressive Growing
   - Sử dụng WGAN-GP loss
   - Thêm Self-Attention layers
   - Feature matching loss
"""