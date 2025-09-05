import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积，轻量化模型的关键。
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

class LightDecoderBlock(nn.Module):
    """
    极致轻量化的解码器块。
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(LightDecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 使用深度可分离卷积代替标准卷积
        self.conv = DepthwiseSeparableConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip_feature):
        x = self.upsample(x)
        x = torch.cat([x, skip_feature], dim=1)
        x = self.conv(x)
        return x

class NanoSeg(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        """
        极致轻量级语义分割模型 NanoSeg
        :param num_classes: 分割的类别数
        :param pretrained: 是否加载 ImageNet 预训练权重
        """
        super(NanoSeg, self).__init__()

        # 1. 加载 MobileNetV3-Small 作为编码器
        mobilenet = mobilenet_v3_small(weights='IMAGENET1K_V1' if pretrained else None)
        self.encoder_features = mobilenet.features

        # 2. 确定用于跳跃连接的层和通道数
        # MobileNetV3-Small 的结构, 需要查看其 `features` 确定
        # H/2 -> [0] -> 16 channels
        # H/4 -> [1] -> 16 channels
        # H/8 -> [3] -> 24 channels
        # H/16 -> [8] -> 48 channels
        skip_channels = [16, 16, 24, 48]
        bottleneck_channels = 576 # 最后一个特征层的通道数

        # 3. 定义解码器 (使用极少的通道数)
        decoder_channels = [64, 48, 32, 16] # 大幅减少通道数

        self.decoder1 = LightDecoderBlock(bottleneck_channels, skip_channels[3], decoder_channels[0])
        self.decoder2 = LightDecoderBlock(decoder_channels[0], skip_channels[2], decoder_channels[1])
        self.decoder3 = LightDecoderBlock(decoder_channels[1], skip_channels[1], decoder_channels[2])
        self.decoder4 = LightDecoderBlock(decoder_channels[2], skip_channels[0], decoder_channels[3])
        
        # 4. 定义分割头
        self.segmentation_head = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)
        
        # 5. 最终上采样
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        skips = []
        # 编码过程
        # 根据 MobileNetV3-Small 的结构索引
        x = self.encoder_features[0](x); skips.append(x)  # H/2
        x = self.encoder_features[1](x); skips.append(x)  # H/4
        x = self.encoder_features[2](x)
        x = self.encoder_features[3](x); skips.append(x)  # H/8
        x = self.encoder_features[4:9](x); skips.append(x) # H/16
        x = self.encoder_features[9:](x) # Bottleneck
        
        # 解码过程
        d1 = self.decoder1(x, skips[3])
        d2 = self.decoder2(d1, skips[2])
        d3 = self.decoder3(d2, skips[1])
        d4 = self.decoder4(d3, skips[0])

        # 分割头
        output = self.segmentation_head(d4)
        output = self.final_upsample(output)

        return output

# --- 模型测试 ---
if __name__ == '__main__':
    # 假设输入图像是 224x224
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 创建模型，假设分割 11 个类别
    model = NanoSeg(num_classes=2, pretrained=True)
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save("model_jit.pt")
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params / 1e6:.2f} M") 
    
    # 前向传播测试
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    # 检查输出尺寸
    print(f"输入尺寸: {dummy_input.shape}")
    print(f"输出尺寸: {output.shape}") # 应为 (2, 11, 224, 224)
    
    assert output.shape == (2, 11, 224, 224)
    print("NanoSeg 模型结构和尺寸检查通过！")
