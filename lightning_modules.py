# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
import albumentations as A
from albumentations.pytorch import ToTensorV2
# 从其他文件导入
from model import NanoSeg
from dataset import Mydataset
# 安装 torchmetrics 用于计算指标
# pip install torchmetrics
from torchmetrics import JaccardIndex
from torchvision.utils import make_grid, save_image 
import torchvision
from loss import EdgeWeightedLoss, DiceLoss


class SegModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters() # 保存超参数，方便后续加载
        
        self.model = NanoSeg(num_classes=num_classes, pretrained=True)
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.ew_loss = EdgeWeightedLoss()
        
        # 初始化 metric: Jaccard Index (IoU)
        self.train_iou = JaccardIndex(task="multiclass", num_classes=num_classes)
        self.val_iou = JaccardIndex(task="multiclass", num_classes=num_classes)
        self.visual_dir = "./visual"
        os.makedirs(self.visual_dir, exist_ok=True)
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss_ce = self.ce_loss(outputs, masks)
        loss_dice = self.dice_loss(outputs, masks)
        loss_ew = self.ew_loss(outputs, masks)
        
        # 组合：0.5 * CE + 0.5 * Dice
        loss = 0.3 * loss_ce + 0.4 * loss_dice + 0.2 * loss_ew
        
        # 计算并记录训练指标
        self.train_iou(outputs, masks)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_iou', self.train_iou, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss_ce = self.ce_loss(outputs, masks)
        loss_dice = self.dice_loss(outputs, masks)
        loss_ew = self.ew_loss(outputs, masks)
        
        # 组合：0.5 * CE + 0.5 * Dice
        loss = 0.3 * loss_ce + 0.4 * loss_dice + 0.2 * loss_ew
        
        # 计算并记录验证指标
        self.val_iou(outputs, masks)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_iou', self.val_iou, on_epoch=True, prog_bar=True)
        # [新增] 仅在每个 Epoch 的第一个 Batch 保存可视化图片
        if batch_idx == 0:
            self.save_validation_visuals(images, masks, outputs)

    def save_validation_visuals(self, images, masks, outputs, num_samples=8):
        """
        保存验证集可视化结果到本地 ./visual 文件夹
        格式：原图 | 真实标签(GT) | 预测结果(Pred)
        """
        # 1. 获取预测类别 (B, C, H, W) -> (B, H, W)
        preds = torch.argmax(outputs, dim=1)
        
        # 2. 限制保存的样本数量 (防止图太大)
        n = min(images.shape[0], num_samples)
        imgs_vis = images[:n].clone() # clone 防止修改原数据
        masks_vis = masks[:n].float()
        preds_vis = preds[:n].float()

        # 3. 反归一化 (Un-normalize)
        # 这里的 mean 和 std 必须与 Albumentations 中定义的一致
        mean = torch.tensor([0.485, 0.456, 0.406]).type_as(imgs_vis).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).type_as(imgs_vis).view(1, 3, 1, 1)
        imgs_vis = imgs_vis * std + mean
        imgs_vis = torch.clamp(imgs_vis, 0, 1) # 限制在 [0, 1] 范围内

        # 4. 处理 Mask 显示
        # 将 Mask 从 (B, H, W) 扩展为 (B, 1, H, W) 并复制为 3 通道 (B, 3, H, W) 以便与 RGB 图像拼接
        # 如果是二分类，0是背景，1是前景。为了显示清楚，我们将 1 映射到 1.0 (白色)
        scale_factor = 1.0 if self.hparams.num_classes <= 2 else (1.0 / (self.hparams.num_classes - 1))
        
        masks_vis = masks_vis.unsqueeze(1) * scale_factor
        preds_vis = preds_vis.unsqueeze(1) * scale_factor
        
        # 转为 3 通道灰度图
        masks_vis = masks_vis.repeat(1, 3, 1, 1)
        preds_vis = preds_vis.repeat(1, 3, 1, 1)

        # 5. 拼接图片: 将 [原图, GT, 预测] 组合在一起
        # stack 之后维度: (N, 3, 3, H, W) -> view -> (N*3, 3, H, W)
        combined = torch.stack([imgs_vis, masks_vis, preds_vis], dim=1) 
        combined = combined.view(-1, 3, imgs_vis.shape[2], imgs_vis.shape[3])

        # 6. 生成网格图 (nrow=3 表示一行显示一组：原图-GT-预测)
        grid = make_grid(combined, nrow=3, padding=2, normalize=False)

        # 7. 保存到本地
        save_path = os.path.join(self.visual_dir, f"epoch_{self.current_epoch:03d}_val.png")
        save_image(grid, save_path)
        # print(f"Saved visualization to {save_path}") # 可选：打印保存信息

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        # 添加学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss", # 监控验证集损失
            },
        }

if __name__ == '__main__':
    # --- 超参数 ---
    NUM_CLASSES = 2
    IMG_SIZE = (320, 320)
    BATCH_SIZE = 100
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 20

    # --- 数据准备 ---
    train_transform, val_transform = A.Compose([
        A.Resize(*IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]), A.Compose([
        A.Resize(*IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    # 使用伪数据集，实际使用时替换成你的真实数据集
    train_dataset =Mydataset(
        bg_image_path="./data/bg_images",
        doc_image_path="/home/lixumin/project/data/question-data/train/images",
        num_samples=BATCH_SIZE*10, # 减少样本数以便快速测试
        transform=train_transform,
        apply_perspective=True,
        perspective_magnitude=0.25,
        neg_sample_prob=0.1,
        doc_only_prob=0.1,
    )
    val_dataset = Mydataset(
        bg_image_path="./data/bg_images",
        doc_image_path="/home/lixumin/project/data/question-data/train/images",
        num_samples=BATCH_SIZE*10, # 减少样本数以便快速测试
        transform=val_transform,
        apply_perspective=True,
        perspective_magnitude=0.25,
        neg_sample_prob=0.1,
        doc_only_prob=0.1,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)
    
    # --- 模型初始化 ---
    model = SegModel(num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE)

    # --- Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        # monitor='val_iou',  # 不再需要监控指标
        dirpath='checkpoints/',
        filename='nanoseg-{epoch:02d}-{val_iou:.2f}', # 建议保留 val_iou 在文件名中，方便后续查看
        save_top_k=-1,  # 修改为 -1，表示保存所有 epoch 的模型
        # mode='max',     # 不再需要模式
        every_n_epochs=1, # 明确指定每1个epoch保存一次，与 save_top_k=-1 效果相同
    )
    # 早停
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
    )
    progress_bar = RichProgressBar()

    # --- 训练器 ---
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='auto', # 自动选择 GPU 或 CPU
        devices='auto',
        callbacks=[checkpoint_callback, early_stopping_callback, progress_bar],
    )

    # --- 开始训练 ---
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # # --- (可选) 导出模型用于部署 ---
    # # 加载最佳模型
    # best_model_path = checkpoint_callback.best_model_path
    # print(f"加载最佳模型: {best_model_path}")
    # trained_model = SegModel.load_from_checkpoint(best_model_path)
    
    # # 导出到 ONNX
    # dummy_input = torch.randn(1, 3, IMG_SIZE[0], IMG_SIZE[1])
    # trained_model.to_onnx("nanoseg.onnx", dummy_input, export_params=True)
    # print("模型已导出到 nanoseg.onnx")