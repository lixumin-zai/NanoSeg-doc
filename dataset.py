import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
import cv2 # 用于图像处理
import random

def find_image_paths_os(root_folder: str) -> list[str]:
    """
    使用 os.walk 递归查找指定文件夹及其子文件夹中所有图片的路径。
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    image_paths = []
    if not os.path.isdir(root_folder):
        print(f"错误: 路径 '{root_folder}' 不是一个有效的文件夹。")
        return []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(image_extensions):
                full_path = os.path.join(dirpath, filename)
                image_paths.append(full_path)
    return image_paths

class Mydataset(Dataset):
    """
    一个用于在线合成文档分割数据的伪数据集。
    它会随机选择背景和文档图片，对文档进行透视变换，然后粘贴到背景上，并生成对应的掩码。
    新增功能：可以按指定概率生成只有背景的负样本，或只有文档（黑背景）的样本。
    """
    def __init__(self, 
        bg_image_path="/path/to/your/bg_images",    
        doc_image_path="/path/to/your/doc_images", # 
        num_samples=1000, # 每个 epoch 生成的样本数量
        transform=None,
        apply_perspective=True,
        perspective_magnitude=0.4, # 默认设为一个较强的值
        neg_sample_prob=0.1, 
        doc_only_prob=0.1   
    ):
        """
        Args:
            bg_image_path (str): 背景图片文件夹路径。
            doc_image_path (str): 文档图片文件夹路径。
            num_samples (int): 每个 epoch 的样本数。
            transform (A.Compose): Albumentations 的 transform pipeline。
            apply_perspective (bool): 是否对文档应用透视变换。
            perspective_magnitude (float): 透视变换的强度，值越大变形越剧烈。
            neg_sample_prob (float): 生成纯背景负样本的概率。
            doc_only_prob (float): 生成纯文档样本的概率。
        """
        self.bg_images = find_image_paths_os(bg_image_path)
        self.doc_images = find_image_paths_os(doc_image_path)
        self.num_samples = num_samples
        self.transform = transform
        self.apply_perspective = apply_perspective
        self.perspective_magnitude = perspective_magnitude
        self.neg_sample_prob = neg_sample_prob
        self.doc_only_prob = doc_only_prob

        # 确保概率设置合理
        assert neg_sample_prob + doc_only_prob < 1.0, \
            "负样本和纯文档样本的概率之和必须小于 1.0"
        
        if not self.bg_images:
            raise ValueError(f"在 '{bg_image_path}' 中没有找到背景图片。")
        if not self.doc_images:
            raise ValueError(f"在 '{doc_image_path}' 中没有找到文档图片。")
        
        print(f"找到 {len(self.bg_images)} 张背景图片和 {len(self.doc_images)} 张文档图片。")
        print(f"样本生成概率 -> 合成图: {1.0 - neg_sample_prob - doc_only_prob:.2f}, "
              f"纯背景: {neg_sample_prob:.2f}, 纯文档: {doc_only_prob:.2f}")

    def __len__(self):
        return self.num_samples

    def _apply_perspective_transform(self, image):
        """
        对单张图片应用随机透视变换。
        """
        h, w = image.shape[:2]
        
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        margin = int(min(h, w) * self.perspective_magnitude)
        if margin <= 0: margin = 1
            
        pt_tl_x = random.randint(0, margin)
        pt_tl_y = random.randint(0, margin)
        pt_tr_x = random.randint(w - margin, w)
        pt_tr_y = random.randint(0, margin)
        pt_br_x = random.randint(w - margin, w)
        pt_br_y = random.randint(h - margin, h)
        pt_bl_x = random.randint(0, margin)
        pt_bl_y = random.randint(h - margin, h)
        
        dst_points = np.float32([
            [pt_tl_x, pt_tl_y],
            [pt_tr_x, pt_tr_y],
            [pt_br_x, pt_br_y],
            [pt_bl_x, pt_bl_y]
        ])
        
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped_image = cv2.warpPerspective(image, M, (w, h), borderValue=(0,0,0)) # 背景填充为黑色
        doc_mask = np.ones((h, w), dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(doc_mask, M, (w, h), borderValue=0)
        
        return warped_image, warped_mask

    def __getitem__(self, idx):
        while True:
            try:
                sample_type = random.random()
                
                # --- CASE 1: 生成负样本 (只有背景) ---
                if sample_type < self.neg_sample_prob:
                    bg_path = random.choice(self.bg_images)
                    composite_img = cv2.cvtColor(cv2.imread(bg_path), cv2.COLOR_BGR2RGB)
                    h, w, _ = composite_img.shape
                    mask = np.zeros((h, w), dtype=np.uint8)

                # --- CASE 2: 生成只有文档的样本 (在黑色背景上) ---
                elif sample_type < self.neg_sample_prob + self.doc_only_prob:
                    doc_path = random.choice(self.doc_images)
                    doc_img = cv2.cvtColor(cv2.imread(doc_path), cv2.COLOR_BGR2RGB)
                    
                    # if self.apply_perspective:
                    #     # 应用变换，结果图自带黑边，正好作为背景
                    #     composite_img, doc_mask_persp = self._apply_perspective_transform(doc_img)
                    #     mask = (doc_mask_persp > 0).astype(np.uint8)
                    # else:
                        # 不应用变换，图像就是文档本身，mask是全1
                    composite_img = doc_img
                    h, w, _ = composite_img.shape
                    mask = np.ones((h, w), dtype=np.uint8)
                
                # --- CASE 3: 生成合成样本 (原始逻辑) ---
                else:
                    # 1. 随机选择背景和文档图片路径
                    bg_path = random.choice(self.bg_images)
                    doc_path = random.choice(self.doc_images)

                    # 2. 读取图片
                    bg_img = cv2.cvtColor(cv2.imread(bg_path), cv2.COLOR_BGR2RGB)
                    doc_img = cv2.cvtColor(cv2.imread(doc_path), cv2.COLOR_BGR2RGB)
                    
                    # 3. 创建一个随机大小的画布
                    target_h = random.randint(doc_img.shape[0], doc_img.shape[0]+300)
                    target_w = random.randint(doc_img.shape[1], doc_img.shape[1]+300)
                    
                    # 4. 将背景图片调整到画布大小
                    canvas = cv2.resize(bg_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    
                    # 5. 随机调整文档图片大小
                    h_doc, w_doc, _ = doc_img.shape
                    scale = random.uniform(0.6, 0.9)
                    
                    if h_doc > w_doc:
                        new_h_doc = int(target_h * scale)
                        new_w_doc = int(w_doc * (new_h_doc / h_doc)) if h_doc > 0 else 0
                    else:
                        new_w_doc = int(target_w * scale)
                        new_h_doc = int(h_doc * (new_w_doc / w_doc)) if w_doc > 0 else 0
                    
                    if new_w_doc <= 0 or new_h_doc <= 0 or new_w_doc > target_w or new_h_doc > target_h:
                        continue
                        
                    doc_resized = cv2.resize(doc_img, (new_w_doc, new_h_doc))

                    # 6. (可选) 应用透视变换
                    if self.apply_perspective:
                        doc_to_paste, doc_paste_mask = self._apply_perspective_transform(doc_resized)
                        doc_paste_mask = (doc_paste_mask > 0).astype(np.uint8)
                    else:
                        doc_to_paste = doc_resized
                        doc_paste_mask = np.ones((new_h_doc, new_w_doc), dtype=np.uint8)

                    # 7. 在画布上随机选择一个粘贴位置
                    max_x = target_w - new_w_doc
                    max_y = target_h - new_h_doc
                    if max_x <= 0 or max_y <= 0: continue
                    x_offset = random.randint(0, max_x)
                    y_offset = random.randint(0, max_y)

                    # 8. 合成图片
                    composite_img = canvas.copy()
                    roi = composite_img[y_offset:y_offset+new_h_doc, x_offset:x_offset+new_w_doc]
                    roi[:] = np.where(doc_paste_mask[..., np.newaxis] == 1, doc_to_paste, roi)

                    # 9. 生成掩码
                    mask = np.zeros((target_h, target_w), dtype=np.uint8)
                    mask_roi = mask[y_offset:y_offset+new_h_doc, x_offset:x_offset+new_w_doc]
                    mask_roi[doc_paste_mask == 1] = 1

                # 10. 应用数据增强 (对所有情况都通用)
                if self.transform:
                    augmented = self.transform(image=composite_img, mask=mask)
                    image = augmented['image']
                    mask = augmented['mask']
                else:
                    image = torch.from_numpy(composite_img.transpose(2, 0, 1)).float() / 255.0
                    mask = torch.from_numpy(mask)

                break # 成功生成样本，跳出while循环
            except Exception as e:
                # 在复杂的数据生成中，偶尔出错是正常的，例如读取到损坏的图片
                # print(f"生成样本时出错: {e}，正在重试...")
                continue
            
        return image, mask.long()

if __name__ == "__main__":
    # 1. 定义 Albumentations 数据增强流程
    transform = A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.RandomBrightnessContrast(p=0.3),
        # A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    # 2. 创建文件夹并提示用户放入图片
    if not os.path.exists("data/bg_images"): os.makedirs("data/bg_images")
    if not os.path.exists("data/doc_images"): os.makedirs("data/doc_images")
    print("请确保在 'data/bg_images' 和 'data/doc_images' 文件夹中放入一些图片用于测试。")

    # 3. 实例化数据集 (使用新增的参数)
    try:
        dataset = Mydataset(
            bg_image_path="./data/bg_images",
            doc_image_path="./data/doc_images",
            num_samples=100,
            transform=transform,
            apply_perspective=True,
            perspective_magnitude=0.25,
            neg_sample_prob=0.1,  # 20% 的概率是纯背景
            doc_only_prob=0.1,  # 20% 的概率是纯文档
        )
    except ValueError as e:
        print(f"无法初始化数据集: {e}")
        dataset = None # 标记数据集创建失败

    if dataset and (not dataset.bg_images or not dataset.doc_images):
        print("测试文件夹为空，无法创建 DataLoader。请添加图片后重试。")
    elif dataset:
        # 4. 创建 DataLoader
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

        # 5. 获取数据并检查
        try:
            images, masks = next(iter(dataloader))
            print(masks)
            print(f"\n成功获取一个批次的数据！")
            print(f"Images batch shape: {images.shape}") # 应为 [8, 3, 512, 512]
            print(f"Images batch dtype: {images.dtype}")
            print(f"Masks batch shape: {masks.shape}")   # 应为 [8, 512, 512]
            print(f"Masks batch dtype: {masks.dtype}")
            
            # 检查这批数据中掩码的唯一值，可以验证是否有负样本
            print("\n检查批次中每个掩码的唯一值:")
            for i in range(images.shape[0]):
                unique_vals = torch.unique(masks[i])
                if len(unique_vals) == 1 and unique_vals[0] == 0:
                    print(f"  样本 {i}: {unique_vals}  <-- 这是一个负样本 (纯背景)")
                elif len(unique_vals) == 1 and unique_vals[0] == 1:
                     print(f"  样本 {i}: {unique_vals}  <-- 这是一个填满画面的纯文档样本")
                else:
                    print(f"  样本 {i}: {unique_vals}  <-- 这是一个合成样本或部分画面的纯文档样本")

            
            # (可选) 可视化第一个样本来验证
            try:
                import matplotlib.pyplot as plt
                
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                
                # 可视化前4个样本
                fig, axes = plt.subplots(4, 2, figsize=(10, 20))
                fig.suptitle("可视化批次中的前4个样本", fontsize=16)

                for i in range(4):
                    img_np = images[i].numpy().transpose(1, 2, 0)
                    img_np = std * img_np + mean
                    img_np = np.clip(img_np, 0, 1)
                    mask_np = masks[i].numpy()

                    # 显示图像
                    axes[i, 0].imshow(img_np)
                    axes[i, 0].set_title(f"样本 {i+1}: 图像")
                    axes[i, 0].axis('off')

                    # 显示掩码
                    axes[i, 1].imshow(mask_np, cmap='gray')
                    axes[i, 1].set_title(f"样本 {i+1}: 掩码 (唯一值: {torch.unique(masks[i]).numpy()})")
                    axes[i, 1].axis('off')

                plt.tight_layout(rect=[0, 0.03, 1, 0.96])
                plt.savefig("./show_synthesized_batch_result.jpg")
                print("\n已生成可视化结果 'show_synthesized_batch_result.jpg'")

            except ImportError:
                print("\nMatplotlib未安装，无法进行可视化。可运行 'pip install matplotlib' 安装。")

        except StopIteration:
            print("DataLoader 为空，无法获取数据。")
        except Exception as e:
            print(f"在获取数据时发生错误: {e}")