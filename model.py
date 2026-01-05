import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from torchvision.models.feature_extraction import create_feature_extractor

class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积: 保持轻量化
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class PPM(nn.Module):
    """
    Pyramid Pooling Module (极简版)
    用于捕获全局上下文信息，解决 Mask 破碎和空洞问题。
    """
    def __init__(self, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1) # 全局平均池化，看清全图
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]
        feat = self.pool(x)
        feat = self.conv(feat)
        feat = F.interpolate(feat, size, mode='bilinear', align_corners=True)
        # 将全局特征加回到局部特征上 (Residual connection)
        return torch.cat([x, feat], dim=1)

class LightDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(LightDecoderBlock, self).__init__()
        # 1. 对 Skip connection 进行降维和处理，滤除高频背景噪声
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels // 2, 1, bias=False), # 降维
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 2. 融合后的卷积
        # 输入通道 = 上一层输出 + 处理后的Skip
        # 这里的 in_channels 是上一层解码器的输出
        self.conv = DepthwiseSeparableConv(in_channels + (out_channels // 2), out_channels)

    def forward(self, x, skip_feature):
        # x: 来自深层 (语义强)
        # skip_feature: 来自浅层 (纹理强)
        
        # 上采样深层特征
        x = self.upsample(x)
        
        # 处理浅层特征 (很重要！减少网格干扰)
        if skip_feature is not None:
            skip_feature = self.skip_conv(skip_feature)
            # 确保尺寸匹配 (处理可能的取整误差)
            if x.shape[2:] != skip_feature.shape[2:]:
                x = F.interpolate(x, size=skip_feature.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip_feature], dim=1)
        else:
            # 如果没有 skip (例如最后一层)，根据维度可能需要调整，但在本架构设计中通常都有
            pass 
            
        x = self.conv(x)
        return x

class NanoSeg(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(NanoSeg, self).__init__()
        
        # 1. 更加稳健的特征提取器
        # MobileNetV3-Small 节点名称:
        # 'features.0': H/2  (16ch)
        # 'features.1': H/4  (16ch)
        # 'features.3': H/8  (24ch)
        # 'features.8': H/16 (48ch)
        # 'features.12': H/32 (576ch) - Bottleneck
        
        backbone = mobilenet_v3_small(weights='IMAGENET1K_V1' if pretrained else None)
        return_nodes = {
            'features.0': 'c1',  # H/2
            'features.1': 'c2',  # H/4
            'features.3': 'c3',  # H/8
            'features.8': 'c4',  # H/16
            'features.12': 'c5', # H/32 (Bottleneck)
        }
        self.backbone = create_feature_extractor(backbone, return_nodes=return_nodes)
        
        # 2. 通道定义
        c1_ch = 16
        c2_ch = 16
        c3_ch = 24
        c4_ch = 48
        c5_ch = 576
        
        decoder_dim = 64 # 统一解码器基准通道
        
        # 3. 上下文模块 (PPM)
        # 将 c5 (576) 压缩并融合全局信息
        self.ppm = PPM(c5_ch, decoder_dim) 
        # PPM 输出通道 = 576 + 64 = 640
        
        # 4. 解码器
        # Layer 4: c5(H/32) -> H/16, 融合 c4
        self.decoder4 = LightDecoderBlock(in_channels=c5_ch + decoder_dim, skip_channels=c4_ch, out_channels=decoder_dim)
        
        # Layer 3: H/16 -> H/8, 融合 c3
        self.decoder3 = LightDecoderBlock(in_channels=decoder_dim, skip_channels=c3_ch, out_channels=decoder_dim)
        
        # Layer 2: H/8 -> H/4, 融合 c2
        self.decoder2 = LightDecoderBlock(in_channels=decoder_dim, skip_channels=c2_ch, out_channels=32)
        
        # Layer 1: H/4 -> H/2, 融合 c1
        self.decoder1 = LightDecoderBlock(in_channels=32, skip_channels=c1_ch, out_channels=16)
        
        # 5. 分割头
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # H/2 -> H/1
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, 1)
        )

    def forward(self, x):
        # 1. 提取特征
        features = self.backbone(x)
        c1, c2, c3, c4, c5 = features['c1'], features['c2'], features['c3'], features['c4'], features['c5']
        
        # 2. Bottleneck + Context
        x = self.ppm(c5) # 增加全局感受野
        
        # 3. 解码
        x = self.decoder4(x, c4)
        x = self.decoder3(x, c3)
        x = self.decoder2(x, c2)
        x = self.decoder1(x, c1)
        
        # 4. 输出
        x = self.final_conv(x)
        
        return x

if __name__ == '__main__':
    dummy_input = torch.randn(2, 3, 320, 320)
    model = NanoSeg(num_classes=2)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"NanoSeg 参数量: {total_params / 1e6:.2f} M")
    
    output = model(dummy_input)
    print(f"Input: {dummy_input.shape}")
    print(f"Output: {output.shape}")
    
    assert output.shape == (2, 2, 320, 320)
    print("测试通过！")