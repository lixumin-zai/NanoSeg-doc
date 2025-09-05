# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import albumentations as A
from albumentations.pytorch import ToTensorV2
# 从其他文件导入
from model import NanoSeg
from dataset import Mydataset
from pytorch_lightning.callbacks import RichProgressBar
# 安装 torchmetrics 用于计算指标
# pip install torchmetrics
from torchmetrics import JaccardIndex

class SegModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters() # 保存超参数，方便后续加载
        
        self.model = NanoSeg(num_classes=num_classes, pretrained=True)
        self.loss_fn = nn.CrossEntropyLoss()
        
        # 初始化 metric: Jaccard Index (IoU)
        self.train_iou = JaccardIndex(task="multiclass", num_classes=num_classes)
        self.val_iou = JaccardIndex(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, masks)
        
        # 计算并记录训练指标
        self.train_iou(outputs, masks)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_iou', self.train_iou, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, masks)
        
        # 计算并记录验证指标
        self.val_iou(outputs, masks)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_iou', self.val_iou, on_epoch=True, prog_bar=True)

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
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 10

    # --- 数据准备 ---
    train_transform, val_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]), A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    # 使用伪数据集，实际使用时替换成你的真实数据集
    train_dataset =Mydataset(
        bg_image_path="./data/bg_images",
        doc_image_path="/home/lixumin/project/data/question-data/train/images",
        num_samples=BATCH_SIZE*100, # 减少样本数以便快速测试
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
    # 保存最好的模型
    checkpoint_callback = ModelCheckpoint(
        monitor='val_iou',
        dirpath='checkpoints/',
        filename='nanoseg-{epoch:02d}-{val_iou:.2f}',
        save_top_k=1,
        mode='max',
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