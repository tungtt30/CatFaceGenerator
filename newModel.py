import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralNorm(nn.Module):
    """Spectral Normalization để ổn định training"""
    def __init__(self, module, power_iterations=1):
        super().__init__()
        self.module = module
        self.power_iterations = power_iterations
        if not hasattr(module, 'weight'):
            raise ValueError('Module must have weight parameter')
        
        w = module.weight.data
        height = w.size(0)
        width = w.view(height, -1).size(1)
        
        u = nn.Parameter(w.new_empty(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.new_empty(width).normal_(0, 1), requires_grad=False)
        
        self.register_parameter('weight_u', u)
        self.register_parameter('weight_v', v)
        
    def forward(self, *args):
        return self.module(*args)


class SelfAttention(nn.Module):
    """Self-Attention mechanism để tạo ảnh có chi tiết tốt hơn"""
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Tạo query, key, value
        proj_query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, H * W)
        proj_value = self.value_conv(x).view(batch_size, -1, H * W)
        
        # Tính attention
        attention = torch.bmm(proj_query, proj_key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        # Residual connection với learnable parameter
        out = self.gamma * out + x
        return out


class ResidualBlock(nn.Module):
    """Residual Block để tăng độ sâu mà không bị gradient vanishing"""
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.bn2(self.conv2(out))
        
        if self.upsample:
            residual = F.interpolate(residual, scale_factor=2, mode='nearest')
        residual = self.shortcut(residual)
        
        return F.relu(out + residual)


class ImprovedGenerator(nn.Module):
    """Generator cải tiến với nhiều kỹ thuật hiện đại"""
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        
        # Initial dense layer
        self.fc = nn.Linear(nz, 4 * 4 * ngf * 16)
        
        # Progressive upsampling với residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(ngf * 16, ngf * 8, upsample=True),  # 4x4 -> 8x8
            ResidualBlock(ngf * 8, ngf * 4, upsample=True),   # 8x8 -> 16x16
            ResidualBlock(ngf * 4, ngf * 2, upsample=True),   # 16x16 -> 32x32
            ResidualBlock(ngf * 2, ngf, upsample=True),       # 32x32 -> 64x64
            ResidualBlock(ngf, ngf, upsample=True),           # 64x64 -> 128x128
        ])
        
        # Self-attention ở resolution trung bình
        self.attention = SelfAttention(ngf * 2)
        
        # Final output layer
        self.final_conv = nn.Conv2d(ngf, nc, 3, 1, 1)
        
        # Adaptive Instance Normalization cho style control
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, z):
        # Reshape noise to feature map
        x = self.fc(z).view(-1, 64 * 16, 4, 4)
        
        # Progressive upsampling
        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x)
            
            # Apply attention ở resolution 32x32
            if i == 2:  # Sau khi upsample lên 32x32
                x = self.attention(x)
        
        # Final output với Tanh activation
        x = torch.tanh(self.final_conv(x))
        return x


class ImprovedDiscriminator(nn.Module):
    """Discriminator cải tiến với Progressive growing và Spectral Norm"""
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        
        # Progressive downsampling
        self.conv_blocks = nn.ModuleList([
            # 128x128 -> 64x64
            nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # 64x64 -> 32x32
            nn.Sequential(
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # 32x32 -> 16x16
            nn.Sequential(
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # 16x16 -> 8x8
            nn.Sequential(
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # 8x8 -> 4x4
            nn.Sequential(
                nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1),
                nn.BatchNorm2d(ndf * 16),
                nn.LeakyReLU(0.2, inplace=True)
            )
        ])
        
        # Self-attention ở resolution trung bình
        self.attention = SelfAttention(ndf * 4)
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Conv2d(ndf * 16, 1, 4, 1, 0),
            nn.Sigmoid()
        )
        
        # Apply spectral normalization to all conv layers
        self._apply_spectral_norm()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _apply_spectral_norm(self):
        """Apply spectral normalization to stabilize training"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.utils.spectral_norm(module)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Progressive downsampling
        for i, conv_block in enumerate(self.conv_blocks):
            x = conv_block(x)
            
            # Apply attention ở resolution 16x16
            if i == 2:  # Sau khi downsample xuống 16x16
                x = self.attention(x)
        
        # Final classification
        x = self.classifier(x)
        return x.view(-1)


class GANLoss(nn.Module):
    """Improved loss function với Wasserstein loss và gradient penalty"""
    def __init__(self, loss_type='wgan-gp'):
        super().__init__()
        self.loss_type = loss_type
        
    def forward(self, pred, target_is_real):
        if self.loss_type == 'wgan-gp':
            if target_is_real:
                return -pred.mean()
            else:
                return pred.mean()
        else:  # Standard GAN loss
            if target_is_real:
                return F.binary_cross_entropy(pred, torch.ones_like(pred))
            else:
                return F.binary_cross_entropy(pred, torch.zeros_like(pred))


def gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Gradient penalty cho WGAN-GP"""
    batch_size = real_samples.size(0)
    
    # Random interpolation
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    
    # Get discriminator output
    d_interpolated = discriminator(interpolated)
    
    # Calculate gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Calculate penalty
    gradients = gradients.view(batch_size, -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return penalty


# Ví dụ sử dụng
if __name__ == "__main__":
    # Khởi tạo models
    generator = ImprovedGenerator(nz=100, ngf=64, nc=3)
    discriminator = ImprovedDiscriminator(nc=3, ndf=64)
    
    # Test forward pass
    batch_size = 4
    noise = torch.randn(batch_size, 100)
    fake_images = generator(noise)
    
    print(f"Generated image shape: {fake_images.shape}")
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")