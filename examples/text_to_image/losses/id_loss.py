# loss.py
import torch.nn as nn
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm  # 用于显示进度条
from losses.insightface_backbone_conv import getarcface
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

import ipdb
st = ipdb.set_trace

def get_affine_crop_matrix(landmarks, image_size=512, crop_size=112, top_expansion_ratio=0.2):
    """
    生成可微分的仿射变换矩阵 theta，用于从 landmarks 提取居中的人脸方形区域。
    
    参数:
        landmarks: Tensor, shape [B, 98, 2]，每张图 98 个关键点
        image_size: int，原图宽高
        crop_size: int，输出裁剪图宽高
        top_expansion_ratio: float，裁剪框上方扩张比例

    返回:
        theta: Tensor, shape [B, 2, 3]，可用于 affine_grid
    """
    # B = batch size
    B = landmarks.shape[0]

    # 1. 计算 landmarks 的边界框（每张图分别取最小最大坐标）
    x_min = landmarks[:, :, 0].min(dim=1).values  # shape [B]
    x_max = landmarks[:, :, 0].max(dim=1).values
    y_min = landmarks[:, :, 1].min(dim=1).values
    y_max = landmarks[:, :, 1].max(dim=1).values

    # 2. 计算宽、高和向上扩展后的 y_min（上额头留空）
    width = x_max - x_min                      # shape [B]
    height = y_max - y_min
    y_min_new = y_min - top_expansion_ratio * height
    height_new = y_max - y_min_new

    # 3. 得到正方形的裁剪框边长 side（选最大边）
    side = torch.max(width, height_new)        # shape [B]

    # 4. 中心点坐标（用于仿射变换）
    cx = (x_min + x_max) / 2                   # [B]
    cy = y_min_new + side / 2                  # [B]

    # 5. 将 center 坐标归一化到 [-1, 1] 坐标系统（供 affine_grid 用）
    scale = crop_size / side                   # 每张图的缩放比例 [B]
    tx = (2 * (cx - image_size / 2) / image_size) * scale  # [B]
    ty = (2 * (cy - image_size / 2) / image_size) * scale  # [B]

    # 6. 构造仿射变换矩阵 theta（完全使用可微操作）
    zeros = torch.zeros_like(scale)            # [B]
    theta = torch.stack([
        torch.stack([scale, zeros, -tx], dim=-1),
        torch.stack([zeros, scale, -ty], dim=-1)
    ], dim=1)  # shape [B, 2, 3]

    return theta

def crop_faces_with_grid_sample(images, landmarks, crop_size=112):
    """
    使用 grid_sample 对 batch 图像做 differentiable 裁剪。
    输入:
        images: [B, 3, H, W] tensor
        landmarks: [B, 98, 2]
    返回:
        crops: [B, 3, crop_size, crop_size]
    """
    B = images.size(0)
    theta = get_affine_crop_matrix(landmarks, image_size=images.shape[2], crop_size=crop_size)  # [B, 2, 3]
    grid = F.affine_grid(theta, size=(B, 3, crop_size, crop_size), align_corners=False)  # [B, H, W, 2]
    crops = F.grid_sample(images, grid, align_corners=False, mode='bilinear', padding_mode='border')  # [B, 3, 112, 112]
    return crops


def extract_arcface_features_from_batch(pred_rgb, coord_list, arcface_model, device):
    """
    全可微实现。注意：arcface_model 必须处于训练模式或开启 grad。
    输入:
        pred_rgb: [B, 3, 512, 512]
        coord_list: [B, 98, 2]
    输出:
        arcface_feats: [B, 512]
    """
    assert pred_rgb.requires_grad, "Input image must require gradients"


    x = pred_rgb.to(device)

    # 可选：取消 CLIP 归一化（可删除）
    # x = un_norm_clip(x) 
    x = torch.nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)  # resize to 256
    x = x[:, :, 35:223, 32:220]  # 粗暴裁剪中央区域 → 大约是人脸区域
    x = torch.nn.functional.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)  # resize to 112

    feats = arcface_model(x)  # 提取特征（不应包含 detach）
    return feats


@torch.no_grad()
def extract_and_save_features_txt(
    image_dir, lmk_txt_dir, save_crop_dir, save_feat_dir, arcface_model, device
):
    transform = T.Compose([
        T.ToTensor(),  # 转为 float32，范围 [0, 1]
    ])
    arcface_model.to(device).eval()

    os.makedirs(save_crop_dir, exist_ok=True)
    os.makedirs(save_feat_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
    image_files.sort()

    for fname in tqdm(image_files, desc="Processing"):
        stem = os.path.splitext(fname)[0]  # 如 0.jpg → 0
        img_path = os.path.join(image_dir, fname)
        lmk_path = os.path.join(lmk_txt_dir, f"{stem}.txt")

        if not os.path.exists(lmk_path):
            print(f"[Skip] No landmark for {fname}")
            continue

        try:
            # 读取图像
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]

            # 读取 landmark txt 文件
            with open(lmk_path, 'r') as f:
                lines = f.readlines()
                landmarks = np.array([[float(x) for x in line.strip().split()] for line in lines])  # shape [98, 2]
                landmarks = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 98, 2]

            # 裁剪人脸区域（可微）
            crops = crop_faces_with_grid_sample(img_tensor, landmarks, crop_size=112)  # [1, 3, 112, 112]
            feat = arcface_model(crops)  # [1, 512]

            # 保存裁剪图
            save_crop_path = os.path.join(save_crop_dir, fname)
            crop_img = T.ToPILImage()(crops.squeeze(0).cpu().clamp(0, 1))
            crop_img.save(save_crop_path)

            # 保存特征
            save_feat_path = os.path.join(save_feat_dir, f"{stem}.npy")
            np.save(save_feat_path, feat.squeeze(0).cpu().numpy())

        except Exception as e:
            print(f"[Error] {fname}: {e}")


def compute_cosine_similarity(feat_dir, num_samples=10):
    """
    随机抽取特征文件，计算彼此之间以及与自身的余弦相似度。
    输入:
        feat_dir: str，特征文件夹路径
        num_samples: int，随机抽取的特征文件数量
    """
    # 获取所有特征文件
    feat_files = [f for f in os.listdir(feat_dir) if f.endswith('.npy')]
    feat_files.sort()

    # 随机抽取 num_samples 个特征文件
    sampled_files = np.random.choice(feat_files, size=num_samples, replace=False)

    # 加载特征
    features = []
    for fname in sampled_files:
        feat_path = os.path.join(feat_dir, fname)
        features.append(np.load(feat_path))  # 加载特征
    features = torch.tensor(features, dtype=torch.float32)  # 转为 Tensor

    # 计算余弦相似度
    cosine_sim_matrix = torch.nn.functional.cosine_similarity(
        features.unsqueeze(1), features.unsqueeze(0), dim=-1
    )  # [num_samples, num_samples]

    # 打印结果
    print("Cosine Similarity Matrix:")
    print("Images:", sampled_files)
    for i in range(num_samples):
        for j in range(num_samples):
            print(f"Cosine similarity between {sampled_files[i]} and {sampled_files[j]}: {cosine_sim_matrix[i, j].item():.4f}")


def crop_resize_and_save_images(image_dir, save_dir, device):
    """
    对文件夹中的图片进行裁剪、resize，并保存到指定文件夹。
    输入:
        image_dir: str，原始图片文件夹路径
        save_dir: str，裁剪后图片保存路径
        device: torch.device，设备（如 GPU 或 CPU）
    """
    transform = T.Compose([
        T.ToTensor(),  # 转为 float32，范围 [0, 1]
    ])
    os.makedirs(save_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
    image_files.sort()

    for fname in tqdm(image_files, desc="Cropping and Resizing Images"):
        img_path = os.path.join(image_dir, fname)
        save_path = os.path.join(save_dir, fname)

        try:
            # 读取图像
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]

            # Resize 到 256x256
            resized_tensor = torch.nn.functional.interpolate(img_tensor, size=(256, 256), mode='bilinear', align_corners=False)

            # 裁剪中央区域
            cropped_tensor = resized_tensor[:, :, 35:223, 32:220]  # [1, 3, 188, 188]

            # Resize 到 112x112
            final_tensor = torch.nn.functional.interpolate(cropped_tensor, size=(112, 112), mode='bilinear', align_corners=False)

            # 转换为 PIL 图像并保存
            final_img = T.ToPILImage()(final_tensor.squeeze(0).cpu().clamp(0, 1))
            final_img.save(save_path)

        except Exception as e:
            print(f"[Error] {fname}: {e}")

@torch.no_grad()
def extract_and_save_features(image_dir, save_feat_dir, arcface_model, device):
    """
    提取裁剪后图片的特征并保存到指定文件夹。
    输入:
        image_dir: str，裁剪后图片文件夹路径
        save_feat_dir: str，特征保存路径
        arcface_model: nn.Module，ArcFace 模型
        device: torch.device，设备（如 GPU 或 CPU）
    """
    transform = T.Compose([
        T.ToTensor(),  # 转为 float32，范围 [0, 1]
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # ArcFace 输入归一化
    ])
    os.makedirs(save_feat_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
    image_files.sort()

    arcface_model.to(device).eval()

    for fname in tqdm(image_files, desc="Extracting Features"):
        img_path = os.path.join(image_dir, fname)
        save_path = os.path.join(save_feat_dir, f"{os.path.splitext(fname)[0]}.npy")

        try:
            # 读取裁剪后的图像
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, 112, 112]

            # 提取特征
            feat = arcface_model(img_tensor)  # [1, 512]

            # 保存特征到文件
            np.save(save_path, feat.squeeze(0).cpu().numpy())

        except Exception as e:
            print(f"[Error] {fname}: {e}")

