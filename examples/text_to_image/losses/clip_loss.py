import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel
from torchvision.transforms import functional as TF
from torchvision import transforms


class CLIPFeatureLoss(nn.Module):
    def __init__(self, clip_model_path: str, device='cuda'):
        super().__init__()
        self.device = device
        self.clip_model = CLIPVisionModel.from_pretrained(
            clip_model_path, local_files_only=True
        ).eval().to(device)
        
        # 冻结 CLIP 权重
        for p in self.clip_model.parameters():
            p.requires_grad = False
        
        # 用于对图像做预处理（适配 CLIP 输入）
        self.clip_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.4815, 0.4578, 0.4082],
                                 std=[0.2686, 0.2613, 0.2758]),
        ])

    def forward(self, pred_img: torch.Tensor, gt_feat: torch.Tensor):
        """
        pred_img: FloatTensor, [B, 3, H, W], 有梯度（输出图）
        gt_feat:  FloatTensor, [B, 257, 1280]，CLIP embedding ground truth（无梯度）
        """
        B = pred_img.shape[0]


        # === 1. 预处理生成图像 ===
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        resized = F.interpolate(pred_img, size=(224, 224), mode='bilinear', align_corners=False)
        normalized = TF.normalize(resized,
                                mean=mean,
                                std=std)

        # 强制 FP32 提高稳定性
        normalized = normalized.to(dtype=torch.float32)


        # === 2. 提取 CLIP embedding（保留梯度）===
        clip_feat = self.clip_model(normalized).last_hidden_state  # [B, 257, 1280]

        # === 3. 分离 CLS 与 Patch 特征 ===
        clip_cls = clip_feat[:, 0]         # [B, 1280]
        clip_patch = clip_feat[:, 1:]      # [B, 256, 1280]
        gt_cls = gt_feat[:, 0]             # [B, 1280]
        gt_patch = gt_feat[:, 1:]          # [B, 256, 1280]

        # === 4. 计算相似度 ===
        cls_sim = F.cosine_similarity(clip_cls, gt_cls, dim=-1)  # [B]
        patch_sim = F.cosine_similarity(clip_patch, gt_patch, dim=-1).mean(dim=-1)  # [B]


        return cls_sim, patch_sim
