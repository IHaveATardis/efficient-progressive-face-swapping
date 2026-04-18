import sys
import os

# 加入 text_to_image 目录为根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

        # # 准备 idnet的ASA控制器
        # if self.idnet is not None:
        #     repeats = 2 if guidance_scale > 1.0 else 1
        #     # 写：特征写入 bank
        #     self.idnet_writer = ReferenceAttentionControl(
        #         self.idnet, mode="write", fusion_blocks="midup", do_classifier_free_guidance=False
        #     )
        #     # 读：主干 UNet 从 bank 读取特征
        #     self.idnet_reader = ReferenceAttentionControl(
        #         self.unet, mode="read", fusion_blocks="midup", 
        #         do_classifier_free_guidance=(guidance_scale > 1.0),
        #         repeats=repeats, style_fidelity=1.0
        #     )
        #     self.idnet_reader.share_bank(self.idnet_writer)
    
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
        eta: float = 0.0):

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

        # idnet_residuals = format_idnet_residuals_for_unet(idnet_residuals)
       
        for i, t in enumerate(timesteps):
            # # expand the latents if we are doing classifier free guidance
            # if do_classifier_free_guidance:
            #     # 条件分支：使用背景条件
            #     cond_latents = torch.cat([latents, bg_latents], dim=1)  # [B, 8, H, W]

            #     # 无条件分支：不使用背景条件，只用原始 latent
            #     uncond_latents = latents  # [B, 4, H, W]

            #     # 拼接两个 batch：注意 cond 是 8 通道，uncond 是 4 通道，不合法！
            #     # 所以需要对 unet 的输入统一通道数（例如 unet 输入必须是 8 通道）

            #     # 正确方式：无条件分支也用 8 通道，但 bg_latents 置零
            #     zero_bg = torch.zeros_like(bg_latents)
            #     uncond_latents = torch.cat([latents, zero_bg], dim=1)

            #     latent_model_input = torch.cat([uncond_latents, cond_latents], dim=0)  # [2B, 8, H, W]
            # else:
            #     latent_model_input = torch.cat([latents, bg_latents], dim=1)
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            fused_latents = latents

            # 对每一步id写入bank
            # print("source_latent grad:",source_latent.requires_grad)
            # idnet_output = None
            # if self.idnet is not None:
            #     idnet_output = self.idnet(source_latent, t, encoder_hidden_states=id_embeddings, return_dict=False)[0]
            #     if idnet_output is not None:
            #         print(f"[DEBUG][T={t}] idnet_output.requires_grad: {idnet_output.requires_grad}")
            #         print(f"[DEBUG][T={t}] idnet_output grad_fn: {idnet_output.grad_fn}")

        
            # 5.3 为当前时间步缩放输入
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
                # noise_pred = self.unet(
                #     latent_model_input,
                #     t,
                #     encoder_hidden_states=encoder_hidden_states,
                #     cross_attention_kwargs={'id_embedding': id_embeddings, 'id_scale': id_scale},
                #     return_dict=False,
                # )[0]
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
            latents = scheduler_output.latents
            log_prob = scheduler_output.log_probs
            # print(f"Latents range after step: min={latents.min().item()}, max={latents.max().item()}")

            all_latents.append(latents)
            all_log_probs.append(log_prob)

            # # 每步结束后清理bank
            # if self.idnet is not None:
            #     self.idnet_writer.clear()
            #     self.idnet_reader.clear()


            # call the callback, if provided
            # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
            #     progress_bar.update()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents
        # print(f"Decoded image range: min={image.min().item()}, max={image.max().item()}")
        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        # print(f"Postprocessed image range: min={image.min().item()}, max={image.max().item()}")

        # if self.idnet is not None:
        #     self.idnet_writer.reset()
        #     self.idnet_reader.reset()


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

        # summarize_idnet_container(idnet_residuals)

        # 5. Denoising loop
        for i, t in enumerate(timesteps):

            # # === 5.1 构造输入 latent（concat 而不是 pixel-wise add）===
            # if guidance_scale > 1.0:
            #     # 条件分支：使用背景条件
            #     cond_latents = torch.cat([latents, bg_latents], dim=1)  # [B, 8, H, W]

            #     # 无条件分支：不使用背景条件，只用原始 latent
            #     uncond_latents = latents  # [B, 4, H, W]

            #     # 拼接两个 batch：注意 cond 是 8 通道，uncond 是 4 通道，不合法！
            #     # 所以需要对 unet 的输入统一通道数（例如 unet 输入必须是 8 通道）

            #     # 正确方式：无条件分支也用 8 通道，但 bg_latents 置零
            #     zero_bg = torch.zeros_like(bg_latents)
            #     uncond_latents = torch.cat([latents, zero_bg], dim=1)

            #     latent_model_input = torch.cat([uncond_latents, cond_latents], dim=0)  # [2B, 8, H, W]
            # else:
            #     latent_model_input = torch.cat([latents, bg_latents], dim=1)

            # # === 5.2 缩放 latent ===
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            fused_latents = latents

            # # 对每一步id写入bank
            # if self.idnet is not None:
            #     _ = self.idnet(source_latent, t, encoder_hidden_states=id_embeddings, return_dict=False)[0]
        
            
            # 5.3 为当前时间步缩放输入
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


            # === 5.4 更新 latents 到上一个时间步 ===
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # 每步结束后清理bank
            # if self.idnet is not None:
            #     self.idnet_writer.clear()
            #     self.idnet_reader.clear()


        # 6. 缩放并解码latents为图像
        latents = 1 / self.vae.config.scaling_factor * latents
        latents = latents.to(dtype=dtype)
        image = self.vae.decode(latents).sample

        # 7. 转换为输出格式
        image = self.image_processor.postprocess(image, output_type=output_type)

        # if self.idnet is not None:
        #     self.idnet_writer.reset()
        #     self.idnet_reader.reset()

        if not return_dict:
            return (image,)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)


