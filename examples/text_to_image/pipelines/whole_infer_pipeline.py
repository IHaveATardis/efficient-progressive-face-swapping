import sys
import os

# 加入 examples/text_to_image 目录为根目录
TEXT_TO_IMAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TEXT_TO_IMAGE_ROOT not in sys.path:
    sys.path.insert(0, TEXT_TO_IMAGE_ROOT)

import torch
from torch import nn
from typing import Optional, Union, Dict, Any
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.models import AutoencoderKL
import torch.utils.checkpoint as checkpoint
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg

from models.idnet.identity_net import IdentityNetModel
from models.idnet.unet_2d_condition import UNet2DConditionModel

import torch.nn.functional as F

from trl.models.modeling_sd_base import DDPOPipelineOutput, scheduler_step
import ipdb
st = ipdb.set_trace

def _iter_tensors(obj, path="root"):
    if torch.is_tensor(obj):
        yield path, obj
    elif isinstance(obj, (list, tuple)):
        for i, x in enumerate(obj):
            yield from _iter_tensors(x, f"{path}[{i}]")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_tensors(v, f"{path}.{k}")
    else:
        # 非张量且非容器，忽略
        return

def _guess_stage_from_NC(N, C):
    res = {4096: "64x64", 1024: "32x32", 256: "16x16", 64: "8x8"}.get(int(N), "?")
    chs = {320: "down0/up3", 640: "down1/up2", 1280: "down2/down3/mid/up0/up1"}.get(int(C), "?")
    return f"{res} / C={C} → {chs}"

def summarize_idnet_container(idnet_residuals, max_items=50):
    print("\n========== [IDNET RESIDUALS SUMMARY] ==========")
    print(f"type: {type(idnet_residuals)}")
    cnt = 0
    for path, t in _iter_tensors(idnet_residuals):
        try:
            shape = tuple(t.shape)
            B = shape[0] if t.ndim >= 1 else None
            N = shape[1] if t.ndim >= 2 else None
            C = shape[-1] if t.ndim >= 1 else None
            stage = " | stage? " + _guess_stage_from_NC(N, C) if (N is not None and C is not None) else ""
            print(f"[{cnt:02d}] {path}: shape={shape}, dtype={t.dtype}, device={t.device}{stage}")
            if t.ndim >= 2:
                with torch.no_grad():
                    _min = float(t.min().item())
                    _max = float(t.max().item())
                    _mean = float(t.mean().item())
                    _std = float(t.std().item())
                print(f"     stats: min={_min:.4f} max={_max:.4f} mean={_mean:.4f} std={_std:.4f}  "
                      f"requires_grad={t.requires_grad} contiguous={t.is_contiguous()}")
        except Exception as e:
            print(f"[{cnt:02d}] {path}: <error reading tensor: {e}>")
        cnt += 1
        if cnt >= max_items:
            print(f"... truncated after {max_items} items ...")
            break
    print("===============================================\n")


class CustomStableDiffusionPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        id_encoder: nn.Module,
        attr_encoder: nn.Module,
        scheduler,
        idnet: Optional[IdentityNetModel] = None,
        image_processor: Optional[VaeImageProcessor] = None,
        requires_safety_checker: bool = False,
        guidance_scale: float = 1.0,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            id_encoder=id_encoder,
            attr_encoder=attr_encoder,
            idnet=idnet
        )
        
        self.image_processor = image_processor or VaeImageProcessor(vae_scale_factor=8)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

       
    # ============ utils ============
    def _prepare_latent_mask(
        self,
        mask: Union[torch.FloatTensor, "PIL.Image.Image"],
        target_hw: tuple[int, int],
        batch_size: int,
        channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.FloatTensor:
        """
        Returns mask in latent space, shape [B, C, H_lat, W_lat], values in [0,1].
        Assumes mask=1 is FACE/EDIT region; background=0 will be preserved.
        """
        if not isinstance(mask, torch.Tensor):
            raise ValueError("mask must be a torch.Tensor of shape [B,1,H,W] or [B,H,W].")
        m = mask
        if m.ndim == 3:
            m = m.unsqueeze(1)                     # [B,1,H,W]
        if m.shape[1] != 1:
            m = m.mean(dim=1, keepdim=True)        # to single channel

        m = m.to(device=device, dtype=dtype)
        H_lat, W_lat = target_hw
        if m.shape[-2:] != (H_lat, W_lat):
            m = F.interpolate(m, size=(H_lat, W_lat), mode="bilinear", align_corners=False)
        m = m.clamp(0.0, 1.0)
        # 轻微羽化，避免边界接缝（不改变整体逻辑）
        m = F.avg_pool2d(m, kernel_size=3, stride=1, padding=1)

        # 扩到 latent 通道数（一般为4）
        if m.shape[0] != batch_size:
            m = m.expand(batch_size, -1, -1, -1)
        m = m.expand(batch_size, channels, H_lat, W_lat)
        return m

    def _ddim_bg_prev_from_x0_and_noise(
        self,
        x0_latents: torch.FloatTensor,           # [B,4,H,W]
        noise: torch.FloatTensor,                # [B,4,H,W]  (same noise used to init x_T)
        t_prev: torch.Tensor,                    # scalar timestep (long)
        alphas_cumprod: torch.FloatTensor,       # [num_train_timesteps]
    ) -> torch.FloatTensor:
        """
        Closed-form background latent at t_prev given x0 and the fixed noise.
        x_{t} = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps
        """
        # 取 alpha_bar[t_prev]
        a_bar = alphas_cumprod[t_prev.long()].to(x0_latents.device)  # scalar tensor
        while a_bar.ndim < x0_latents.ndim:
            a_bar = a_bar.view([1] * (x0_latents.ndim - 1) + [1])
        # a_bar: [1,1,1,1]
        return torch.sqrt(a_bar) * x0_latents + torch.sqrt(1.0 - a_bar) * noise
    
    def _pipeline_step_with_grad(self,
        source_id_embedding: torch.Tensor,
        clip_embedding: torch.Tensor,
        bg_image: Union[torch.FloatTensor, "PIL.Image.Image"],
        src_image: Optional[Union[torch.FloatTensor, "PIL.Image.Image"]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        id_scale: float = 1.0,
        gradient_checkpoint: bool = True,
        backprop_strategy: str = 'gaussian',
        backprop_kwargs: Dict[str, Any] = None,
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        device: Optional[torch.device] = None,
        guidance_rescale: float = 0.0,
        eta: float = 0.0,
        mask: Optional[torch.FloatTensor] = None,   # 新增：脸部掩膜（1=脸）
    ):

        # device = self._execution_device
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = self.unet.dtype

        
        backprop_timestep = -1
    
        while backprop_timestep >= num_inference_steps or backprop_timestep < 1:    
            if backprop_strategy == 'gaussian':
                backprop_timestep = int(torch.distributions.Normal(backprop_kwargs['mean'], backprop_kwargs['std']).sample().item())
            elif backprop_strategy == 'uniform':
                backprop_timestep = int(torch.randint(backprop_kwargs['min'], backprop_kwargs['max'], (1,)).item())
            elif backprop_strategy == 'fixed':
                backprop_timestep = int(backprop_kwargs['value'])
        
        # 1. 预处理并编码背景图像
        bg_image = self.image_processor.preprocess(bg_image).to(device=device, dtype=dtype)
        bg_latents = self.vae.encode(bg_image).latent_dist.sample()
        bg_latents = bg_latents * self.vae.config.scaling_factor

        # 1.2 编码源图像
        if self.idnet is not None:
            # 把 src_image 转 latent
            source_img = self.image_processor.preprocess(src_image).to(device=device, dtype=dtype)
            source_latent = self.vae.encode(source_img).latent_dist.sample() * self.vae.config.scaling_factor
            source_latent.requires_grad_(True)  # 确保构图
            # print("[DEBUG] source_latent.requires_grad:", source_latent.requires_grad)


        # 2. 准备初始噪声
        batch_size = bg_latents.shape[0]
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, bg_latents.shape[2], bg_latents.shape[3]),
            generator=generator,
            device=device,
            dtype=dtype
        )
        init_noise = latents.clone()  # 保存初始噪声，用于 closed-form 背景轨迹

        # 3. 设置时间步
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        do_classifier_free_guidance = guidance_scale > 1.0

        # 4. 生成条件嵌入
        encoder_hidden_states = self.attr_encoder(
            clip_embedding.to(device=device, dtype=dtype)
        ).to(dtype=dtype)

        id_embeddings = self.id_encoder(
            source_id_embedding.to(device=device, dtype=dtype)
        ).to(dtype=dtype)

        # 5. 预备 mask & alphas
        use_mask_bg_lock = mask is not None
        if use_mask_bg_lock:
            M_lat = self._prepare_latent_mask(
                mask=mask,
                target_hw=(bg_latents.shape[2], bg_latents.shape[3]),
                batch_size=batch_size,
                channels=latents.shape[1],
                device=device,
                dtype=dtype,
            )
        # alphas_cumprod: 用于 closed-form 背景轨迹
        if not hasattr(self.scheduler, "alphas_cumprod"):
            raise AttributeError("scheduler must have `alphas_cumprod` for masked background replacement.")
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device=device, dtype=dtype)

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        all_latents = [latents]
        all_log_probs = []

        idnet_residuals = ()

        if self.idnet is not None:
            idnet_timestep = torch.zeros_like(timesteps[0])

            # 获取idnet输出
            idnet_sample, idnet_residuals = self.idnet(
                sample = source_latent,
                timestep=idnet_timestep,
                encoder_hidden_states=id_embeddings,
                return_dict=False,
            )
       
        for i, t in enumerate(timesteps):
            fused_latents = latents
            # 为当前时间步缩放输入
            latent_model_input = self.scheduler.scale_model_input(fused_latents, t)

            # predict the noise residual
            if gradient_checkpoint:
                noise_pred = checkpoint.checkpoint(
                    self.unet,
                    latent_model_input,
                    t,
                    encoder_hidden_states,
                    cross_attention_kwargs={'id_embedding': id_embeddings, 'id_scale': id_scale},
                    use_reentrant=False,
                    idnet_residuals=idnet_residuals,
                )[0]
            else:
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs={'id_embedding': id_embeddings, 'id_scale': id_scale},
                    return_dict=False,
                    idnet_residuals=idnet_residuals,
                )[0]
            

            if i < backprop_timestep:
                noise_pred = noise_pred.detach()

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://huggingface.co/papers/2305.08891
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            # print(f"Noise prediction range: min={noise_pred.min().item()}, max={noise_pred.max().item()}")

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_output = scheduler_step(self.scheduler, noise_pred, t, latents, eta)
            latents_pred = scheduler_output.latents  # 预测得到的 x_{t-1}^{pred}
            log_prob = scheduler_output.log_probs

             # === 逐步背景复写：用 x_{t-1}^{bg-ref} 替换背景（mask=0） ===
            if use_mask_bg_lock:
                # 定义 t_prev（下一循环所对应的时刻）；最后一步取 0
                if i == len(timesteps) - 1:
                    t_prev = torch.tensor(0, device=timesteps.device, dtype=timesteps.dtype)
                else:
                    t_prev = timesteps[i + 1]

                x_bg_prev = self._ddim_bg_prev_from_x0_and_noise(
                    x0_latents=bg_latents,
                    noise=init_noise,
                    t_prev=t_prev,
                    alphas_cumprod=alphas_cumprod,
                )
                latents = M_lat * latents_pred + (1.0 - M_lat) * x_bg_prev.to(dtype=dtype)
            else:
                latents = latents_pred
           
            all_latents.append(latents)
            all_log_probs.append(log_prob)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        if not return_dict:
            return (image,)

        return DDPOPipelineOutput(image, all_latents, all_log_probs)

    
    def rgb_with_grad(self, *args, **kwargs) -> DDPOPipelineOutput:
        return self._pipeline_step_with_grad(*args, **kwargs)


    @torch.no_grad()
    def __call__(
        self,
        source_id_embedding: torch.Tensor,
        clip_embedding: torch.Tensor,
        bg_image: Union[torch.FloatTensor, "PIL.Image.Image"],
        src_image: Optional[Union[torch.FloatTensor, "PIL.Image.Image"]] = None,
        id_scale: float = 1.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        device: Optional[torch.device] = None,
        mask: Optional[torch.FloatTensor] = None,   # 新增：脸部掩膜（1=脸）
    ):
        # device = self._execution_device
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = self.unet.dtype

        # 1. 预处理并编码背景图像
        bg_image = self.image_processor.preprocess(bg_image).to(device=device, dtype=dtype)
        bg_latents = self.vae.encode(bg_image).latent_dist.sample()
        bg_latents = bg_latents * self.vae.config.scaling_factor

        # 1.2 编码源图像
        if self.idnet is not None:
            # 把 src_image 转 latent
            source_img = self.image_processor.preprocess(src_image).to(device=device, dtype=dtype)
            source_latent = self.vae.encode(source_img).latent_dist.sample() * self.vae.config.scaling_factor


        # 2. 准备初始噪声
        batch_size = bg_latents.shape[0]
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, bg_latents.shape[2], bg_latents.shape[3]),
            generator=generator,
            device=device,
            dtype=dtype
        )
        init_noise = latents.clone()  # 保存初始噪声，用于 closed-form 背景轨迹

        # 3. 设置时间步
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. 生成条件嵌入
        encoder_hidden_states = self.attr_encoder(
            clip_embedding.to(device=device, dtype=dtype)
        ).to(dtype=dtype)

        id_embeddings = self.id_encoder(
            source_id_embedding.to(device=device, dtype=dtype)
        ).to(dtype=dtype)

        idnet_residuals = ()

        if self.idnet is not None:
            idnet_timestep = torch.zeros_like(timesteps[0])

            # 获取idnet输出
            idnet_sample, idnet_residuals = self.idnet(
                sample = source_latent,
                timestep=idnet_timestep,
                encoder_hidden_states=id_embeddings,
                return_dict=False,
            )

        # 5. 预备 mask & alphas
        use_mask_bg_lock = mask is not None
        if use_mask_bg_lock:
            M_lat = self._prepare_latent_mask(
                mask=mask,
                target_hw=(bg_latents.shape[2], bg_latents.shape[3]),
                batch_size=batch_size,
                channels=latents.shape[1],
                device=device,
                dtype=dtype,
            )
        if not hasattr(self.scheduler, "alphas_cumprod"):
            raise AttributeError("scheduler must have `alphas_cumprod` for masked background replacement.")
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device=device, dtype=dtype)


        # 6. Denoising loop
        for i, t in enumerate(timesteps):
            fused_latents = latents
            # 为当前时间步缩放输入
            latent_model_input = self.scheduler.scale_model_input(fused_latents, t)


            # === 5.3 预测噪声 ===
            if guidance_scale > 1.0:
                # 复制encoder_hidden_states用于unconditional branch
                encoder_hidden_states_expanded = torch.cat([encoder_hidden_states] * 2)
                id_embeddings_expanded = torch.cat([id_embeddings] * 2)
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states_expanded,
                    cross_attention_kwargs={'id_embedding': id_embeddings_expanded, 'id_scale': id_scale},
                    idnet_residuals=idnet_residuals,
                ).sample.to(dtype=dtype)
                
                # 执行guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs={'id_embedding': id_embeddings, 'id_scale': id_scale},
                    idnet_residuals=idnet_residuals,
                ).sample.to(dtype=dtype)


            # === 更新到上一个时间步（预测值） ===
            latents_pred = self.scheduler.step(noise_pred, t, latents).prev_sample

            # === 逐步背景复写：用 x_{t-1}^{bg-ref} 替换背景（mask=0） ===
            if use_mask_bg_lock:
                if i == len(timesteps) - 1:
                    t_prev = torch.tensor(0, device=timesteps.device, dtype=timesteps.dtype)
                else:
                    t_prev = timesteps[i + 1]

                x_bg_prev = self._ddim_bg_prev_from_x0_and_noise(
                    x0_latents=bg_latents,
                    noise=init_noise,
                    t_prev=t_prev,
                    alphas_cumprod=alphas_cumprod,
                )
                latents = M_lat * latents_pred + (1.0 - M_lat) * x_bg_prev.to(dtype=dtype)
            else:
                latents = latents_pred


        # 6. 缩放并解码latents为图像
        latents = 1 / self.vae.config.scaling_factor * latents
        latents = latents.to(dtype=dtype)
        image = self.vae.decode(latents).sample

        # 7. 转换为输出格式
        image = self.image_processor.postprocess(image, output_type=output_type)


        if not return_dict:
            return (image,)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)


