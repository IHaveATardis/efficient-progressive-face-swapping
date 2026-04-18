#!/usr/bin/env python
# coding=utf-8
                                                              
                                
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""
import sys
import os

                              
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import logging
import math
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import datasets
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


import diffusers
from diffusers import UniPCMultistepScheduler
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from components.id_encoder import ID2Token
from components.attr_encoder import Image2Token

from components.attention_processor import AttnProcessor2_0 as AttnProcessor
from components.attention_processor import IDAttnProcessor2_0 as IDAttnProcessor

from data.multi_domain_pair_dataset import DomainSpec, SingleCelebAPairDataset, SingleFFHQPairDataset
from data.multi_domain_pair_dataset import collate_fn as _orig_collate_fn

import ipdb
st = ipdb.set_trace

if is_wandb_available():
    import wandb

                                                                                            
check_min_version("0.33.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def collate_fn(examples):
    """
    在不改变原有 collate 行为的前提下，附加一份元信息列表：
    batch["__meta__"] = [{"tgt_dom":..., "tgt_stem":..., "src_dom":..., "src_stem":...}, ...]
    """
    batch = _orig_collate_fn(examples)             
    metas = []
    for ex in examples:
                            
        metas.append({
            "tgt_dom":  str(ex.get("dataset_tgt", "UNK")),
            "tgt_stem": str(ex.get("stem_tgt", "UNK")),
            "src_dom":  str(ex.get("dataset_src", "UNK")),
            "src_stem": str(ex.get("stem_src", "UNK")),
        })
    batch["__meta__"] = metas                                 
    return batch

def parse_args():
    import argparse, os

    parser = argparse.ArgumentParser(description="Training script (multi-domain face swap).")

                          
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)

    parser.add_argument("--output_dir", type=str, default="sd-model-finetuned-lora")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true", default=False)
    parser.add_argument("--random_flip", action="store_true", default=False)

    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help='["linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup"]')
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--snr_gamma", type=float, default=None)

    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)

    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--local_rank", type=int, default=-1)

    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--noise_offset", type=float, default=0.0)

    parser.add_argument("--rank", type=int, default=4, help="LoRA rank")

                    
    parser.add_argument("--id_scale", type=float, default=1.0)
    parser.add_argument("--id_loss_weight", type=float, default=0.5)
    parser.add_argument("--mse_loss_weight", type=float, default=0.5)
    parser.add_argument("--clip_cls_weight", type=float, default=0.1)
    parser.add_argument("--clip_patch_weight", type=float, default=0.1)
    parser.add_argument("--prediction_type", type=str, default=None)

                      
    parser.add_argument("--no_load_optimizer", action="store_true")

                                  
    parser.add_argument("--arcface_model_ckpt", type=str, required=True)
    parser.add_argument("--clip_model_ckpt", type=str, required=True)

                                                  
    parser.add_argument("--dataset_name", type=str, choices=["celeba", "ffhq"], required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--arcface_dir", type=str, required=True)
    parser.add_argument("--whole_arc_dir", type=str, required=True)
    parser.add_argument("--clip_dir", type=str, required=True)
    parser.add_argument("--lmk_dir", type=str, default=None)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--train_count", type=int, default=None)
    parser.add_argument("--test_count", type=int, default=2000)

              
    parser.add_argument("--avoid_same_identity", action="store_true")
    parser.add_argument("--same_id_cos_thr", type=float, default=0.8)

                            
    args = parser.parse_args()

                       
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

                                         
    def _check_dir(p, name, required=True):
        if p is None:
            if required:
                raise ValueError(f"--{name} is required.")
            return
        if not os.path.isdir(p):
            raise ValueError(f"--{name} not found or not a directory: {p}")

    _check_dir(args.image_dir, "image_dir")
    _check_dir(args.arcface_dir, "arcface_dir")
    _check_dir(args.whole_arc_dir, "whole_arc_dir")
    _check_dir(args.clip_dir, "clip_dir")
    _check_dir(args.lmk_dir, "lmk_dir", required=False)
    _check_dir(args.mask_dir, "mask_dir", required=False)

    if args.train_count is None:
        args.train_count = 28000 if args.dataset_name == "celeba" else 68000

                       
    if not os.path.exists(args.arcface_model_ckpt):
        raise ValueError(f"--arcface_model_ckpt not found: {args.arcface_model_ckpt}")
    if not os.path.exists(args.clip_model_ckpt):
        raise ValueError(f"--clip_model_ckpt not found: {args.clip_model_ckpt}")

    return args



def hack_unet_attn_layers(unet,lora_r=64, lora_alpha=64):
    id_adapter_attn_procs = {}
    for name, _ in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is not None:
            id_adapter_attn_procs[name] = IDAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
            ).to(unet.device)
        else:
            id_adapter_attn_procs[name] = AttnProcessor()
    unet.set_attn_processor(id_adapter_attn_procs)
    return nn.ModuleList(unet.attn_processors.values())


def main():
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )
    

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    print("CUDA Available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

                          
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

                                                                         
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

                                                 
    if args.seed is not None:
        set_seed(args.seed)

                                    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
                                           
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

                                                  
    hack_unet_attn_layers(unet, lora_r=args.rank, lora_alpha=args.rank)

                                                     
    unet.requires_grad_(False)
    vae.requires_grad_(False)

                  
    id_encoder = ID2Token(
        id_dim=512, 
        text_hidden_size=768, 
        max_length=77, 
        num_layers=3
    ).to(accelerator.device)

                     
    attr_encoder = Image2Token(
        visual_hidden_size=1280, 
        text_hidden_size=768,
        max_length=77,
        num_layers=3
    ).to(accelerator.device)

             
    id_encoder.train()
    attr_encoder.train()


                                                                                                                                     
                                                                                                      
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

                                                                        
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

                                                           
                                                             
                               
                                             
                                                                                                                  
                         
                                                                                                              
                                                                                  
                                                                                                                                                           
                                           
                                                                                                                                                             
                                            
                                                                                                                                                  
                                                                                  
                                                                                                                                                         
                                                                            
                                                                

                                                                    
    unet.add_adapter(unet_lora_config)

                                                    
    for name, processor in unet.attn_processors.items():
        if isinstance(processor, torch.nn.Module):
            if hasattr(processor, "id_to_k"):
                for param in processor.id_to_k.parameters():
                    param.requires_grad_(True)
            if hasattr(processor, "id_to_v"):
                for param in processor.id_to_v.parameters():
                    param.requires_grad_(True)

    if args.mixed_precision == "fp16":
                                                           
        cast_training_params(unet, dtype=torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

                                                     
                                                                                              
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

                              
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    lora_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))
    params_to_optimize = (
        lora_layers
        + list(id_encoder.parameters())
        + list(attr_encoder.parameters())
    )


    optimizer = optimizer_cls(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    logger.info(f"LoRA layers trainable parameters: {sum(p.numel() for p in lora_layers if p.requires_grad)}")
    logger.info(f"IDEncoder trainable parameters: {sum(p.numel() for p in id_encoder.parameters() if p.requires_grad)}")
    logger.info(f"AttrEncoder trainable parameters: {sum(p.numel() for p in attr_encoder.parameters() if p.requires_grad)}")
   
    logger.info(f"Total trainable parameters: {sum(p.numel() for p in params_to_optimize if p.requires_grad)}")


                                 
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

                                  
    domain = DomainSpec(
        name=args.dataset_name,
        image_dir=args.image_dir,
        arcface_dir=args.arcface_dir,
        whole_arcface_dir=args.whole_arc_dir,
        clip_dir=args.clip_dir,
        lmk_dir=args.lmk_dir,
        mask_dir=args.mask_dir,
        train_count=args.train_count,
        test_count=args.test_count,
        group=args.dataset_name,
    )


    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            unet_lora_layers_to_save = None
            id_encoder_to_save = None
            attr_encoder_to_save = None
            unet_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(unet))):
                    unet_lora_layers_to_save = get_peft_model_state_dict(model)
                    unet_to_save = model.state_dict()                 
                elif isinstance(model, type(unwrap_model(id_encoder))):
                    id_encoder_to_save = model.state_dict()
                elif isinstance(model, type(unwrap_model(attr_encoder))):
                    attr_encoder_to_save = model.state_dict()
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                                                                                        
                weights.pop()

                               
            StableDiffusionPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=convert_state_dict_to_diffusers(unet_lora_layers_to_save),
                safe_serialization=True,
            )

                                       
            if id_encoder_to_save is not None:
                torch.save(id_encoder_to_save, os.path.join(output_dir, "id_encoder.pt"))
            if attr_encoder_to_save is not None:
                torch.save(attr_encoder_to_save, os.path.join(output_dir, "attr_encoder.pt"))
                           
            if unet_to_save is not None:
                torch.save(unet_to_save, os.path.join(output_dir, "unet.pt"))


    def load_model_hook(models, input_dir):
        unet_ = None
        id_encoder_ = None
        attr_encoder_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(unwrap_model(id_encoder))):
                id_encoder_ = model
            elif isinstance(model, type(unwrap_model(attr_encoder))):
                attr_encoder_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

                           
        lora_state_dict = StableDiffusionPipeline.lora_state_dict(input_dir)[0]
        unet_state_dict = {
            f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")
        }

        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")

        if incompatible_keys is not None:
                                            
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

                                   
        if id_encoder_ is not None:
            id_encoder_path = os.path.join(input_dir, "id_encoder.pt")
            if os.path.exists(id_encoder_path):
                state_dict_token = torch.load(id_encoder_path)
                if any(k.startswith("module.") for k in state_dict_token.keys()):
                    state_dict_token = {k.replace("module.", ""): v for k, v in state_dict_token.items()}
                id_encoder_.load_state_dict(state_dict_token)

        if attr_encoder_ is not None:
            attr_encoder_path = os.path.join(input_dir, "attr_encoder.pt")
            if os.path.exists(attr_encoder_path):
                state_dict_attr = torch.load(attr_encoder_path)
                if any(k.startswith("module.") for k in state_dict_attr.keys()):
                    state_dict_attr = {k.replace("module.", ""): v for k, v in state_dict_attr.items()}
                attr_encoder_.load_state_dict(state_dict_attr)


                                                       
        if args.mixed_precision == "fp16":
            models = [unet_]
            if id_encoder_ is not None:
                models.append(id_encoder_)
            if attr_encoder_ is not None:
                models.append(attr_encoder_)
            cast_training_params(models)


    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)


    with accelerator.main_process_first():
        if args.dataset_name == "ffhq":
            train_dataset = SingleFFHQPairDataset(
                domain=domain,
                split="train",
                transform=train_transforms,
                mode="inpaint",
                pair_mode="random",
                self_pair_prob=1.0,
                avoid_same_identity=args.avoid_same_identity,
                same_id_cos_thr=args.same_id_cos_thr,
                seed=args.seed,
            )
        else:
            train_dataset = SingleCelebAPairDataset(
                domain=domain,
                split="train",
                transform=train_transforms,
                mode="inpaint",
                pair_mode="random",
                self_pair_prob=1.0,
                avoid_same_identity=args.avoid_same_identity,
                same_id_cos_thr=args.same_id_cos_thr,
                seed=args.seed,
            )

                                           
    train_dataset.set_epoch(0)

                                             
    if accelerator.is_main_process:
        n_total = len(train_dataset)
        n_train_imgs = len(train_dataset.items) if hasattr(train_dataset, "items") else n_total

        print(f"[DATA:{args.dataset_name}] train(images)={n_train_imgs}")
        print(f"[DATA] total targets per epoch (index_map) = {n_total}")

                                     
        gpus = accelerator.num_processes
        bs_per_gpu = args.train_batch_size
        global_bs = bs_per_gpu * gpus * args.gradient_accumulation_steps

                               
        steps_per_epoch_per_gpu = math.ceil(n_total / (bs_per_gpu * gpus))
        print(f"[DATA] gpus={gpus}, bs_per_gpu={bs_per_gpu}, global_bs={global_bs}")
        print(f"[DATA] expected steps/epoch(per GPU) = {steps_per_epoch_per_gpu}")
        print(f"[DATA] expected total steps = {steps_per_epoch_per_gpu * args.num_train_epochs}")


                                                                     
    def seed_worker(worker_id):
        import os
        worker_info = torch.utils.data.get_worker_info()
        ds = worker_info.dataset
        rank = int(os.environ.get("RANK", "0"))
        base = ds.base_seed + 1000 * ds.epoch + 10_000 * rank
        ds.rng = random.Random(base + worker_id)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,                       
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.dataloader_num_workers > 0),
        worker_init_fn=seed_worker,
    )

                                                             
                                                                                               
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

                                                
    unet, optimizer, train_dataloader, lr_scheduler, id_encoder, attr_encoder = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler, id_encoder, attr_encoder
    )

                                                                                                              
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
                                                             
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

                                                                                  
                                                                 
    if accelerator.is_main_process:
        if args.report_to == "wandb":
            config = {}
            for k, v in vars(args).items():
                if isinstance(v, (int, float, str, bool, torch.Tensor)):
                    config[k] = v
                else:
                    print(f"Skipping {k} of type {type(v)} for config")
                    
            accelerator.init_trackers(project_name="correctdata-faceswap-stage1", config=config)

            
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

                                                                     
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
                                            
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            ckpt_dir = os.path.join(args.output_dir, path)

            if args.no_load_optimizer:
                           
                load_model_hook([unet, id_encoder, attr_encoder], ckpt_dir)

                global_step = 0
                first_epoch = 0
                initial_global_step = global_step
            else:
                                               
                accelerator.load_state(ckpt_dir)
                global_step = int(path.split("-")[1])
                initial_global_step = global_step
                first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
                                                          
        disable=not accelerator.is_local_main_process,
    )

                                          
    sample_log_limit = 2000                        
    sample_log_count = 0
    sample_log_buffer = []

            
    sample_log_dir = os.path.join(args.output_dir, "train_logs")
    os.makedirs(sample_log_dir, exist_ok=True)
    sample_log_path = os.path.join(sample_log_dir, "sampling_log.txt")
    with open(sample_log_path, "w", encoding="utf-8") as f:
        f.write("# batch_index, local_iter, global_step, rank, items=[(tgt_dom:tgt_stem <- src_dom:src_stem) ...]\n")


    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
                                          
        train_dataset.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
                                                     
            if accelerator.is_main_process and sample_log_count < sample_log_limit:
                metas = batch.get("__meta__", None)
                if metas is not None:
                    pairs_str = " ".join(
                        f"[{m['tgt_dom']}:{m['tgt_stem']} <- {m['src_dom']}:{m['src_stem']}]"
                        for m in metas
                    )
                    sample_log_buffer.append(
                        f"{sample_log_count},{step},{global_step},{accelerator.process_index},{pairs_str}\n"
                    )
                    sample_log_count += 1
                    if (sample_log_count % 20 == 0) or (sample_log_count == sample_log_limit):
                        try:
                            with open(sample_log_path, "a", encoding="utf-8") as f:
                                f.writelines(sample_log_buffer)
                            sample_log_buffer.clear()
                        except Exception as e:
                            logger.warning(f"[sampling_log] write failed: {e}")
            with accelerator.accumulate(unet):
                                                
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                                                            
                noise = torch.randn_like(latents)
                if args.noise_offset:
                                                                                 
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]
                                                         
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                                                                                            
                                                         
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                                                         
                encoder_hidden_states = attr_encoder(
                    batch["clip_embedding"].to(accelerator.device)
                ).to(dtype=weight_dtype)

                                                       
                id_embeddings = id_encoder(
                    batch["source_id_embeddings"].to(accelerator.device)
                ).to(dtype=weight_dtype)

                cross_attention_kwargs={'id_embedding': id_embeddings, 'id_scale': args.id_scale}

                                                                          
                if args.prediction_type is not None:
                                                                 
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                                            
                model_pred = unet(
                    noisy_latents, 
                    timesteps, 
                    encoder_hidden_states, 
                    cross_attention_kwargs=cross_attention_kwargs, 
                    return_dict=False
                )[0]

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                                                                                                  
                                                                                                              
                                                                         
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                                                                                                      
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                               
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                                                                                            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                                                                   
                if accelerator.is_main_process:
                                   
                    csv_root = os.path.join(args.output_dir, "train_logs")
                    os.makedirs(csv_root, exist_ok=True)
                    csv_path = os.path.join(csv_root, "loss.csv")
                    if not os.path.exists(csv_path):
                        with open(csv_path, "w") as f:
                            f.write("step,loss\n")
                    with open(csv_path, "a") as f:
                        f.write(f"{global_step},{train_loss:.6f}\n")

                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                                                                                                                   
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                                                                                                                    
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

                          
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)

        accelerator.save_state(args.output_dir)


    accelerator.end_training()


if __name__ == "__main__":
    main()