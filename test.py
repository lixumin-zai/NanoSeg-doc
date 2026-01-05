import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import argparse

# 确保 SegModel 类被导入，因为 load_from_checkpoint 需要它
from lightning_modules import SegModel 

def predict_image(model, image_path, transform, device, num_classes):
    """
    对单张图片进行推理并返回预测的掩码。

    Args:
        model (pl.LightningModule): 加载好的 PyTorch Lightning 模型。
        image_path (str): 输入图片的路径。
        transform (A.Compose): 应用于图片的 Albumentations 变换。
        device (torch.device): 'cuda' or 'cpu'。
        num_classes (int): 类别数。

    Returns:
        tuple: 包含原始图片(RGB), 预测掩码, 和叠加后的图片。
    """
    # 1. 加载和预处理图片
    # cv2 默认以 BGR 格式读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"图片未找到: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 转换为 RGB
    original_h, original_w = image.shape[:2]

    # 应用和训练时一样的变换（除了数据增强部分）
    transformed = transform(image=image_rgb)
    input_tensor = transformed['image'].unsqueeze(0).to(device) # 增加 batch 维度并移动到设备

    # 2. 模型推理
    model.eval() # 设置为评估模式
    with torch.no_grad(): # 关闭梯度计算，节省内存和计算资源
        output = model(input_tensor)

    # 3. 后处理
    # output 的形状是 [1, num_classes, H, W]
    # 我们需要在类别维度上找到最大值的索引来得到预测的类别
    pred_mask = torch.argmax(output, dim=1).squeeze(0) # 移除 batch 和 channel 维度
    pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)

    # 4. 将掩码缩放回原始尺寸
    pred_mask_resized = cv2.resize(
        pred_mask_np, 
        (original_w, original_h), 
        interpolation=cv2.INTER_NEAREST # 必须使用最近邻插值，以保持类别标签的离散性
    )

    return image_rgb, pred_mask_resized

def visualize(original_image, mask, num_classes):
    """
    将原始图片、掩码和叠加结果可视化。
    """
    # 定义一个颜色映射，为每个类别分配一种颜色
    # 类别0 (背景) -> 黑色 (不显示)
    # 类别1 (文档) -> 绿色
    # 你可以为更多类别添加更多颜色
    colors = np.array([
        [0, 0, 0],       # 背景 (类别 0)
        [0, 255, 0],     # 文档 (类别 1)
        # [255, 0, 0],   # 如果有类别 2
        # ...
    ], dtype=np.uint8)

    # 创建一个彩色的掩码
    color_mask = colors[mask]

    # 将彩色掩码与原始图片混合
    # alpha 是透明度，0.5 表示 50% 的透明度
    overlay = cv2.addWeighted(original_image, 1, color_mask, 0.5, 0)

    # 使用 matplotlib 显示结果
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(color_mask)
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay Result')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("./result.jpg")


def export_onnx():
    trained_model = SegModel.load_from_checkpoint("/home/lixumin/project/nanoseg/checkpointsv1/nanoseg-epoch=03-val_iou=0.99.ckpt")
    # # 导出到 ONNX
    dummy_input = torch.randn(1, 3, 320, 320)
    trained_model.to_onnx("nanoseg.onnx", dummy_input, export_params=True)
    print("模型已导出到 nanoseg.onnx")


if __name__ == '__main__':
    # --- 设置命令行参数 ---
    parser = argparse.ArgumentParser(description="Inference script for NanoSeg model.")
    parser.add_argument('--checkpoint', default="/home/lixumin/project/Enet/checkpoints/nanoseg-epoch=09-val_iou=0.97.ckpt", help='Path to the model checkpoint (.ckpt) file.')
    parser.add_argument('--image', default="/home/lixumin/project/Enet/test_images/1ebc3f7b-48d7-11f0-8616-c1c656c74d25.jpg", help='Path to the input image.')
    args = parser.parse_args()

    # --- 配置 ---
    NUM_CLASSES = 2 # 必须与训练时一致
    IMG_SIZE = (320, 320)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 定义推理时用的数据变换 ---
    # 注意：这里我们使用和验证集相同的变换，不包含随机翻转等数据增强
    inference_transform = A.Compose([
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # --- 加载模型 ---
    print(f"Loading model from checkpoint: {args.checkpoint}")
    # 使用 SegModel.load_from_checkpoint 加载，它会自动恢复模型结构、权重和超参数
    try:
        model = SegModel.load_from_checkpoint(args.checkpoint, map_location=DEVICE)
        model.to(DEVICE)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure that 'train.py' (containing the SegModel class) is in the same directory.")
        exit()

    # --- 执行推理和可视化 ---
    print(f"Running inference on image: {args.image}")
    original_image, prediction_mask = predict_image(
        model=model,
        image_path=args.image,
        transform=inference_transform,
        device=DEVICE,
        num_classes=NUM_CLASSES
    )
    
    print("Visualizing results...")
    visualize(original_image, prediction_mask, NUM_CLASSES)