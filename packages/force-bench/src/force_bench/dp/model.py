from __future__ import annotations

import copy
import math

import torch
from torch import nn
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torchvision.models import ResNet18_Weights, resnet18


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)

    def forward(self, timestep):
        timestep = timestep.float()
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(torch.arange(half, device=timestep.device) * -scale)
        embedding = timestep[:, None] * frequencies[None]
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)
        return F.pad(embedding, (0, self.dim - embedding.shape[-1]))


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, groups=8):
        super().__init__()
        padding = kernel_size // 2
        group_count = min(groups, out_channels)
        while out_channels % group_count:
            group_count -= 1
        self.net = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size,
                                           padding=padding),
                                 nn.GroupNorm(group_count, out_channels), nn.Mish())

    def forward(self, value):
        return self.net(value)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, condition_dim, kernel_size=5,
                 groups=8, predict_scale=True):
        super().__init__()
        self.out_channels, self.predict_scale = out_channels, predict_scale
        self.blocks = nn.ModuleList((
            Conv1DBlock(in_channels, out_channels, kernel_size, groups),
            Conv1DBlock(out_channels, out_channels, kernel_size, groups)))
        self.condition = nn.Sequential(nn.Mish(), nn.Linear(
            condition_dim, out_channels * (2 if predict_scale else 1)))
        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, value, condition):
        output = self.blocks[0](value)
        modulation = self.condition(condition).unsqueeze(-1)
        if self.predict_scale:
            scale, bias = modulation.reshape(
                modulation.shape[0], 2, self.out_channels, 1).unbind(1)
            output = scale * output + bias
        else:
            output = output + modulation
        return self.blocks[1](output) + self.residual(value)


class ConditionalUnet1D(nn.Module):
    """Official-style global-condition 1D U-Net over ``[B,T,action_dim]``."""
    def __init__(self, action_dim, global_condition_dim, diffusion_embed_dim=256,
                 down_dims=(256, 512, 1024), kernel_size=5):
        super().__init__()
        dimensions = [action_dim, *down_dims]
        pairs = list(zip(dimensions[:-1], dimensions[1:]))
        self.time_encoder = nn.Sequential(
            SinusoidalPositionEmbedding(diffusion_embed_dim),
            nn.Linear(diffusion_embed_dim, diffusion_embed_dim * 4), nn.Mish(),
            nn.Linear(diffusion_embed_dim * 4, diffusion_embed_dim))
        condition_dim = diffusion_embed_dim + global_condition_dim
        block = lambda a, b: ConditionalResidualBlock1D(
            a, b, condition_dim, kernel_size=kernel_size, predict_scale=True)
        self.down = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(pairs):
            self.down.append(nn.ModuleList((block(dim_in, dim_out), block(dim_out, dim_out),
                nn.Conv1d(dim_out, dim_out, 3, stride=2, padding=1)
                if index < len(pairs) - 1 else nn.Identity())))
        middle = dimensions[-1]
        self.middle = nn.ModuleList((block(middle, middle), block(middle, middle)))
        self.up = nn.ModuleList()
        for dim_in, dim_out in reversed(pairs[1:]):
            self.up.append(nn.ModuleList((block(dim_out * 2, dim_in), block(dim_in, dim_in),
                nn.ConvTranspose1d(dim_in, dim_in, 4, stride=2, padding=1))))
        self.final = nn.Sequential(Conv1DBlock(down_dims[0], down_dims[0], kernel_size),
                                   nn.Conv1d(down_dims[0], action_dim, 1))

    def forward(self, sample, timestep, global_condition):
        value = sample.transpose(1, 2)
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=value.device, dtype=torch.long)
        if timestep.ndim == 0:
            timestep = timestep[None]
        timestep = timestep.to(value.device).expand(value.shape[0])
        condition = torch.cat((self.time_encoder(timestep), global_condition), dim=-1)
        skips = []
        for first, second, downsample in self.down:
            value = second(first(value, condition), condition)
            skips.append(value)
            value = downsample(value)
        for layer in self.middle:
            value = layer(value, condition)
        for first, second, upsample in self.up:
            skip = skips.pop()
            if value.shape[-1] != skip.shape[-1]:
                value = F.interpolate(value, size=skip.shape[-1], mode="nearest")
            value = upsample(second(first(torch.cat((value, skip), dim=1), condition), condition))
        return self.final(value).transpose(1, 2)


class ImageEncoder(nn.Module):
    def __init__(self, resize=224, crop=0, pretrained=False, imagenet_normalize=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        network = resnet18(weights=weights)
        network.fc = nn.Identity()
        for name, module in list(network.named_modules()):
            if isinstance(module, nn.BatchNorm2d):
                parent = network
                parts = name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], nn.GroupNorm(max(1, module.num_features // 16),
                                                        module.num_features))
        self.network, self.resize, self.crop = network, int(resize), int(crop)
        self.imagenet_normalize = bool(imagenet_normalize)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, images):
        images = images.float() / 255.0 if images.dtype == torch.uint8 else images.float()
        if self.resize:
            images = F.interpolate(images, size=(self.resize, self.resize), mode="bilinear",
                                   align_corners=False)
        if self.crop:
            height, width = images.shape[-2:]
            top, left = (height - self.crop) // 2, (width - self.crop) // 2
            images = images[..., top:top + self.crop, left:left + self.crop]
        if self.imagenet_normalize:
            images = (images - self.mean) / self.std
        return self.network(images)


class DiffusionPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.camera_names = tuple(config.camera_names)
        self.vision = ImageEncoder(config.image_resize, config.image_crop,
                                   config.vision_pretrained,
                                   config.image_imagenet_normalize)
        per_observation = len(self.camera_names) * config.vision_feature_dim + config.proprio_dim
        condition_dim = config.observation_horizon * per_observation
        self.unet = ConditionalUnet1D(config.action_dim, condition_dim,
                                      config.diffusion_embed_dim, config.unet_dims)
        self.scheduler = DDPMScheduler(
            num_train_timesteps=config.diffusion_train_timesteps,
            beta_schedule="squaredcos_cap_v2", prediction_type="epsilon", clip_sample=True)
        self.register_buffer("proprio_mean", torch.zeros(config.proprio_dim))
        self.register_buffer("proprio_std", torch.ones(config.proprio_dim))
        self.register_buffer("action_min", -torch.ones(config.action_dim))
        self.register_buffer("action_max", torch.ones(config.action_dim))

    def set_normalization(self, statistics):
        for name in ("proprio_mean", "proprio_std", "action_min", "action_max"):
            value = torch.as_tensor(statistics[name], dtype=getattr(self, name).dtype,
                                    device=getattr(self, name).device)
            if value.shape != getattr(self, name).shape:
                raise ValueError(f"normalization {name} has shape {tuple(value.shape)}")
            getattr(self, name).copy_(value)

    def normalization_state(self):
        return {name: getattr(self, name).detach().cpu().numpy()
                for name in ("proprio_mean", "proprio_std", "action_min", "action_max")}

    def normalize_actions(self, actions):
        span = (self.action_max - self.action_min).clamp_min(1e-6)
        return 2.0 * (actions - self.action_min) / span - 1.0

    def unnormalize_actions(self, actions):
        span = (self.action_max - self.action_min).clamp_min(1e-6)
        return (actions + 1.0) * 0.5 * span + self.action_min

    def encode_observation(self, proprio, images):
        if proprio.shape[1:] != (self.config.observation_horizon, self.config.proprio_dim):
            raise ValueError("proprio must be [B,observation_horizon,18]")
        features = []
        for name in self.camera_names:
            image = images[name]
            if image.shape[:3] != (proprio.shape[0], self.config.observation_horizon, 3):
                raise ValueError(f"{name} RGB must be [B,observation_horizon,3,H,W]")
            flat = image.reshape(-1, *image.shape[2:])
            features.append(self.vision(flat).reshape(proprio.shape[0],
                                                       self.config.observation_horizon, -1))
        normalized_proprio = (proprio - self.proprio_mean) / self.proprio_std.clamp_min(1e-6)
        features.append(normalized_proprio)
        return torch.cat(features, dim=-1).flatten(1)

    def compute_loss(self, proprio, images, actions):
        if actions.shape[1:] != (self.config.action_horizon, self.config.action_dim):
            raise ValueError("actions must be [B,action_horizon,9]")
        clean = self.normalize_actions(actions)
        noise = torch.randn_like(clean)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps,
                                  (clean.shape[0],), device=clean.device)
        noisy = self.scheduler.add_noise(clean, noise, timesteps)
        prediction = self.unet(noisy, timesteps,
                               self.encode_observation(proprio, images))
        return F.mse_loss(prediction, noise)

    @torch.no_grad()
    def predict_actions(self, proprio, images, *, generator=None, inference_steps=None):
        condition = self.encode_observation(proprio, images)
        sample = torch.randn((proprio.shape[0], self.config.action_horizon,
                              self.config.action_dim), device=proprio.device,
                             dtype=proprio.dtype, generator=generator)
        steps = int(inference_steps or self.config.inference_steps)
        self.scheduler.set_timesteps(steps, device=proprio.device)
        for timestep in self.scheduler.timesteps:
            predicted_noise = self.unet(sample, timestep, condition)
            sample = self.scheduler.step(predicted_noise, timestep, sample,
                                         generator=generator).prev_sample
        return self.unnormalize_actions(sample)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model).eval()
        self.shadow.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for target, source in zip(self.shadow.parameters(), model.parameters()):
            target.lerp_(source.detach(), 1.0 - self.decay)
        for target, source in zip(self.shadow.buffers(), model.buffers()):
            target.copy_(source)
