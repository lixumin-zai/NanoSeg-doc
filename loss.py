import os
import cv2
import numpy as np
import torch
import torch.nn as nn


class EdgeWeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none') # 必须设为 none 以便加权

    def forward(self, pred, target):
        # pred: (B, 2, H, W) -> 需要取前景通道
        # target: (B, H, W)
        
        # 1. 提取 target 的边缘
        target_np = target.cpu().numpy().astype(np.uint8)
        edge_weight = torch.ones_like(target).float()
        
        for i in range(target.shape[0]):
            # 对每个样本做 Canny 边缘检测 (target 是 0/1，乘255)
            edges = cv2.Canny(target_np[i] * 255, 100, 200)
            # 对边缘区域进行膨胀，增加容错范围
            edges = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=1)
            # 转回 tensor
            edge_mask = torch.from_numpy(edges).to(target.device) > 0
            # 2. 设置权重：边缘处的 Loss 权重设为 5 倍
            edge_weight[i][edge_mask] = 5.0 

        # 3. 计算 BCE Loss
        # 假设 pred 是 2 通道，取第 1 通道(前景) 与 target 做二分类损失
        pred_logits = pred[:, 1, :, :] # 取前景 logits
        target_float = target.float()
        
        loss = self.bce(pred_logits, target_float)
        
        # 4. 应用权重
        loss = (loss * edge_weight).mean()
        
        return loss

# 需要先定义或导入 DiceLoss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: (B, C, H, W) -> softmax后
        # target: (B, H, W)
        pred = torch.softmax(pred, dim=1)
        
        # 取前景类 (假设 class 1 是文档)
        pred_flat = pred[:, 1].contiguous().view(-1)
        
        # 将 target 转为 one-hot 或 float (假设 target 只有 0, 1)
        target_flat = target.contiguous().view(-1).float()

        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice