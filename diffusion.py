import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import math
from tqdm import tqdm
import random


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
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
        
        print(f"Loaded {len(self.images)} images from {root_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


class SinusoidalPositionEmbeddings(nn.Module):
    """Position embeddings cho timesteps trong diffusion process"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block với time embedding"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.block1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.block2(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        self.group_norm = nn.GroupNorm(8, channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1)
        self.to_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.group_norm(x)
        
        qkv = self.to_qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        k = k.permute(0, 2, 3, 1).view(b, h * w, c)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)
        
        attention = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(c), dim=-1)
        out = attention @ v
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)
        
        return self.to_out(out) + x


class UNet(nn.Module):
    """U-Net architecture cho diffusion model"""
    def __init__(self, in_channels=3, model_channels=128, out_channels=3, 
                 num_res_blocks=2, attention_resolutions=[8, 16], 
                 dropout=0.1, channel_mult=[1, 2, 2, 2], num_heads=4):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Input projection
        self.input_proj = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Encoder (downsampling)
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        input_block_chans = [ch]
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)]
                ch = mult * model_channels
                
                # Add attention at specified resolutions
                if ch // model_channels in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                
                self.down_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            
            # Downsample (except last level)
            if level < len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                input_block_chans.append(ch)
        
        # Middle block
        self.middle_block = nn.ModuleList([
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch),
            ResidualBlock(ch, ch, time_embed_dim, dropout)
        ])
        
        # Decoder (upsampling)
        self.up_blocks = nn.ModuleList()
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [ResidualBlock(ch + input_block_chans.pop(), 
                                     mult * model_channels, time_embed_dim, dropout)]
                ch = mult * model_channels
                
                # Add attention at specified resolutions
                if ch // model_channels in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                
                # Upsample (except last block)
                if level > 0 and i == num_res_blocks:
                    layers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
                
                self.up_blocks.append(nn.ModuleList(layers))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1)
        )
    
    def forward(self, x, timesteps):
        # Time embedding
        time_emb = self.time_embed(timesteps)
        
        # Input projection
        h = self.input_proj(x)
        
        # Store skip connections
        hs = [h]
        
        # Encoder
        for layers in self.down_blocks:
            for layer in layers:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, time_emb)
                else:
                    h = layer(h)
            hs.append(h)
        
        # Middle block
        for layer in self.middle_block:
            if isinstance(layer, ResidualBlock):
                h = layer(h, time_emb)
            else:
                h = layer(h)
        
        # Decoder
        for layers in self.up_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in layers:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, time_emb)
                else:
                    h = layer(h)
        
        # Output projection
        return self.output_proj(h)


class DiffusionScheduler:
    """Scheduler cho diffusion process"""
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and reverse process
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1 - self.alpha_cumprod_prev) / (1 - self.alpha_cumprod)
    
    def add_noise(self, original_samples, noise, timesteps):
        """Add noise to original samples"""
        sqrt_alpha_prod = self.sqrt_alpha_cumprod[timesteps].flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alpha_cumprod[timesteps].flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def sample_prev_timestep(self, model_output, timestep, sample):
        """Sample from reverse process"""
        pred_original_sample = (sample - self.sqrt_one_minus_alpha_cumprod[timestep] * model_output) / self.sqrt_alpha_cumprod[timestep]
        
        # Clip for stability
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # Get variance
        variance = self.posterior_variance[timestep]
        
        if timestep > 0:
            noise = torch.randn_like(sample)
        else:
            noise = torch.zeros_like(sample)
        
        # Compute x_{t-1}
        pred_prev_sample = (
            self.sqrt_alpha_cumprod[timestep - 1] * pred_original_sample +
            torch.sqrt(variance) * noise
        )
        
        return pred_prev_sample


class DiffusionTrainer:
    """Class để quản lý training diffusion model"""
    
    def __init__(self, data_root, image_size=64, batch_size=32, lr=1e-4, 
                 num_epochs=500, num_timesteps=1000, save_every=50):
        
        self.data_root = data_root
        self.image_size = image_size
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.num_timesteps = num_timesteps
        self.save_every = save_every
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Setup data
        self.setup_data()
        
        # Create model and scheduler
        self.model = UNet(
            in_channels=3,
            model_channels=128,
            out_channels=3,
            num_res_blocks=2,
            attention_resolutions=[8, 16],
            channel_mult=[1, 2, 3, 4],
            dropout=0.1
        ).to(self.device)
        
        self.scheduler = DiffusionScheduler(num_timesteps=num_timesteps)
        
        # Move scheduler tensors to device
        for attr_name in ['betas', 'alphas', 'alpha_cumprod', 'alpha_cumprod_prev', 
                         'sqrt_alpha_cumprod', 'sqrt_one_minus_alpha_cumprod', 'posterior_variance']:
            setattr(self.scheduler, attr_name, getattr(self.scheduler, attr_name).to(self.device))
        
        # Setup optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Setup loss
        self.criterion = nn.MSELoss()
        
        # Track losses
        self.losses = []
        
        print("Diffusion model initialized successfully!")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_data(self):
        """Setup data loading với augmentation"""
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])
        
        dataset = DogDataset(self.data_root, transform=transform)
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"Dataset: {len(dataset)} images, {len(self.dataloader)} batches per epoch")
    
    def train_step(self, batch):
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Sample timesteps
        timesteps = torch.randint(0, self.num_timesteps, (batch.size(0),), device=self.device)
        
        # Sample noise
        noise = torch.randn_like(batch)
        
        # Add noise to images
        noisy_images = self.scheduler.add_noise(batch, noise, timesteps)
        
        # Predict noise
        noise_pred = self.model(noisy_images, timesteps)
        
        # Calculate loss
        loss = self.criterion(noise_pred, noise)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def sample_images(self, num_samples=16):
        """Generate sample images"""
        self.model.eval()
        
        with torch.no_grad():
            # Start with random noise
            samples = torch.randn(num_samples, 3, self.image_size, self.image_size, device=self.device)
            
            # Reverse diffusion process
            for t in tqdm(range(self.num_timesteps - 1, -1, -1), desc="Sampling"):
                timesteps = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
                
                # Predict noise
                noise_pred = self.model(samples, timesteps)
                
                # Remove noise
                samples = self.scheduler.sample_prev_timestep(noise_pred, t, samples)
        
        self.model.train()
        return samples
    
    def save_samples(self, epoch, num_samples=16):
        """Save sample images"""
        samples = self.sample_images(num_samples)
        
        # Create directory
        os.makedirs('diffusion_samples', exist_ok=True)
        
        # Save images
        vutils.save_image(
            samples,
            f'diffusion_samples/samples_epoch_{epoch:03d}.png',
            normalize=True,
            value_range=(-1, 1),
            nrow=4
        )
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        os.makedirs('diffusion_checkpoints', exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'scheduler_params': {
                'num_timesteps': self.scheduler.num_timesteps,
                'betas': self.scheduler.betas.cpu(),
            }
        }
        
        torch.save(checkpoint, f'diffusion_checkpoints/checkpoint_epoch_{epoch}.pth')
    
    def train(self):
        """Main training loop"""
        print("Starting Diffusion Training...")
        
        self.model.train()
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
            for batch in pbar:
                batch = batch.to(self.device)
                
                # Training step
                loss = self.train_step(batch)
                
                epoch_loss += loss
                num_batches += 1
                self.losses.append(loss)
                
                # Update progress bar
                pbar.set_postfix({'Loss': f'{loss:.4f}'})
            
            # Calculate average loss
            avg_loss = epoch_loss / num_batches
            print(f"Epoch [{epoch+1}/{self.num_epochs}] - Average Loss: {avg_loss:.4f}")
            
            # Save samples and checkpoint
            if (epoch + 1) % self.save_every == 0:
                print("Generating samples...")
                self.save_samples(epoch + 1)
                self.save_checkpoint(epoch + 1)
        
        print("Training completed!")
        
        # Final checkpoint and samples
        self.save_checkpoint(self.num_epochs)
        self.save_samples(self.num_epochs, num_samples=64)
        
        # Plot losses
        self.plot_losses()
    
    def plot_losses(self):
        """Plot training losses"""
        plt.figure(figsize=(10, 6))
        
        # Smooth the losses for better visualization
        window_size = 100
        if len(self.losses) > window_size:
            smoothed_losses = []
            for i in range(window_size, len(self.losses)):
                smoothed_losses.append(np.mean(self.losses[i-window_size:i]))
            plt.plot(smoothed_losses, label='Smoothed Loss')
        
        plt.plot(self.losses, alpha=0.3, label='Raw Loss')
        plt.title("Diffusion Model Training Loss")
        plt.xlabel("Training Steps")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig('diffusion_training_loss.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_final_samples(self, num_samples=64):
        """Generate final high-quality samples"""
        print(f"Generating {num_samples} final samples...")
        
        all_samples = []
        batch_size = 8  # Generate in smaller batches to save memory
        
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            samples = self.sample_images(current_batch_size)
            all_samples.append(samples.cpu())
        
        all_samples = torch.cat(all_samples, dim=0)
        
        # Save final samples
        vutils.save_image(
            all_samples,
            'final_diffusion_dogs.png',
            normalize=True,
            value_range=(-1, 1),
            nrow=8
        )
        
        return all_samples


def main():
    """Main function để chạy training"""
    
    # Set random seed
    set_seed(42)
    
    # Đường dẫn đến folder chứa ảnh chó
    data_root = "data/train/d"  # Thay đổi path này
    
    # Tạo trainer với hyperparameters tối ưu
    trainer = DiffusionTrainer(
        data_root=data_root,
        image_size=64,
        batch_size=32,          # Batch size nhỏ hơn vì diffusion cần nhiều memory
        lr=1e-4,                # Learning rate cho diffusion
        num_epochs=1,         # Diffusion cần train lâu hơn GAN
        num_timesteps=1000,     # Số timesteps trong diffusion process
        save_every=25           # Save samples mỗi 25 epochs
    )
    
    # Bắt đầu training
    trainer.train()
    
    # Generate final samples
    print("Generating final high-quality samples...")
    trainer.generate_final_samples(64)


def load_and_generate(checkpoint_path, num_samples=16):
    """Load trained model và generate ảnh mới"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = UNet(
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=[8, 16],
        channel_mult=[1, 2, 3, 4],
        dropout=0.1
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create scheduler
    scheduler = DiffusionScheduler(num_timesteps=1000)
    
    # Move scheduler tensors to device
    for attr_name in ['betas', 'alphas', 'alpha_cumprod', 'alpha_cumprod_prev', 
                     'sqrt_alpha_cumprod', 'sqrt_one_minus_alpha_cumprod', 'posterior_variance']:
        setattr(scheduler, attr_name, getattr(scheduler, attr_name).to(device))
    
    # Generate samples
    with torch.no_grad():
        samples = torch.randn(num_samples, 3, 64, 64, device=device)
        
        for t in tqdm(range(999, -1, -1), desc="Generating"):
            timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
            noise_pred = model(samples, timesteps)
            samples = scheduler.sample_prev_timestep(noise_pred, t, samples)
    
    # Save generated images
    vutils.save_image(
        samples,
        'generated_dogs_from_checkpoint.png',
        normalize=True,
        value_range=(-1, 1),
        nrow=4
    )
    
    return samples


if __name__ == "__main__":
    main()


# === TRAINING TIPS FOR DIFFUSION MODELS ===
"""
TIPS CHO TRAINING DIFFUSION MODEL VỚI 20K ẢNH CHÓ:

1. ADVANTAGES CỦA DIFFUSION SO VỚI GAN:
   - Không bị mode collapse
   - Training ổn định hơn
   - Chất lượng ảnh tốt hơn
   - Không cần balance Generator/Discriminator

2. HYPERPARAMETER TUNING:
   - Learning rate: 1e-4 (thấp hơn GAN)
   - Batch size: 8-16 (cần nhiều GPU memory)
   - Epochs: 300-500 (train lâu hơn GAN)
   - Timesteps: 1000 (standard)

3. MEMORY OPTIMIZATION:
   - Sử dụng gradient checkpointing nếu thiếu memory
   - Mixed precision training (FP16)
   - Batch size nhỏ hơn

4. QUALITY MONITORING:
   - Check samples mỗi 25-50 epochs
   - Loss giảm dần và ổn định
   - Không có oscillation như GAN

5. SAMPLING TIPS:
   - DDIM sampling nhanh hơn DDPM
   - Có thể giảm timesteps khi inference (50-100 steps)
   - Classifier-free guidance để tăng chất lượng

6. EXPECTED RESULTS:
   - Epoch 50-100: Ảnh bắt đầu có hình dạng
   - Epoch 150-200: Chi tiết rõ ràng
   - Epoch 250+: Chất lượng cao, realistic
"""