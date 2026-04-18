import torch
from torch import nn
from typing import Optional, Union
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel


import ipdb
st = ipdb.set_trace

class CustomStableDiffusionPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        id_encoder: nn.Module,
        attr_encoder: nn.Module,
        scheduler,
        image_processor: Optional[VaeImageProcessor] = None,
        requires_safety_checker: bool = False,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            id_encoder=id_encoder,
            attr_encoder=attr_encoder,
        )
        
        self.image_processor = image_processor or VaeImageProcessor(vae_scale_factor=8)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    @torch.no_grad()
    def __call__(
        self,
        source_id_embedding: torch.Tensor,
        clip_embedding: torch.Tensor,
        bg_image: Union[torch.FloatTensor, "PIL.Image.Image"],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        id_scale: float = 1.0,
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

        # 5. Denoising loop
        for i, t in enumerate(timesteps):
            # 5.1 准备条件和无条件分支的输入
            if guidance_scale > 1.0:
                # 无条件分支只使用latents
                uncond_latents = latents
                # 条件分支使用latents和bg_latents的组合
                cond_latents = torch.cat([latents, bg_latents], dim=1)
                latent_model_input = torch.cat([uncond_latents, cond_latents], dim=0)
            else:
                # 非CFG模式，直接组合latents和bg_latents
                latent_model_input = torch.cat([latents, bg_latents], dim=1)
            
            
            # 5.3 为当前时间步缩放输入
            latent_model_input = self.scheduler.scale_model_input(latents, t)

            # 5.4 预测噪声残差
            if guidance_scale > 1.0:
                # 复制encoder_hidden_states用于unconditional branch
                encoder_hidden_states_expanded = torch.cat([encoder_hidden_states] * 2)
                id_embeddings_expanded = torch.cat([id_embeddings] * 2)
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states_expanded,
                    cross_attention_kwargs={'id_embedding': id_embeddings_expanded, 'id_scale': id_scale},
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
                ).sample.to(dtype=dtype)

            # 5.5 计算前一个时间步的样本
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 6. 缩放并解码latents为图像
        latents = 1 / self.vae.config.scaling_factor * latents
        latents = latents.to(dtype=dtype)
        image = self.vae.decode(latents).sample

        # 7. 转换为输出格式
        image = self.image_processor.postprocess(image, output_type=output_type)

        if not return_dict:
            return (image,)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)


