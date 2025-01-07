import logging
import math
import types
from copy import deepcopy
from functools import partial
from typing import Dict, Literal, Optional, Union

import torch
import torch.nn as nn
from gsk_configlib import instantiate_object
from timm.models.vision_transformer import Block

from aiml_cell_imaging.models.aggregation import get_aggregation_layer
from aiml_cell_imaging.models.modules import (
    ContextPatchEmbed,
    FlipPatchEmbed,
    RandomPatchDropout,
    RoPEBlock,
)
from aiml_cell_imaging.models.pos_embed import build_2d_sinusoidal_pos_embed
from aiml_cell_imaging.models.pretrained import (
    FOUNDATION_MODELS,
    freeze_parameters,
    get_imagenet_feature_extractor,
    get_named_model,
    map_channels_to_rbg,
)



class MaskedEncoder(nn.Module):
    """
    Encoder part of masked autoencoder

    Three recommended choices of MaskedEncoder:
    1. base:
        (embed_dim, depth, num_heads) = (768, 12, 12)
    2. large:
        (embed_dim, depth, num_heads) = (1024, 24, 16)
    3. huge:
        (embed_dim, depth, num_heads) = (1280, 32, 16)

    The default setting uses large ViT encoder

    Parameters
    ----------
    img_size : int
        input image size
    patch_size : int
        patch size
    in_chans : int
        number of input channels
    mask_ratio : float
        percentage of patches that are randomly masked (droped) during training
    sync_mask : bool
        if true, the same random patche dropout is done across all channels.
        Otherwise, random patch dropout is performed independently for each channel.
    expects_dict : bool, optional
        model accepts a dictionary as inputs containing images and channel indices
        default is True
    pretrained : bool, optional
        load pretrained weights from HuggingFace model if pretrained=True
    num_classes : Optional[int], optional
        number of classes for classification head, used in finetuning
    global_pool : Literal["avg", "token"], optional
        type of global pooling for final sequence
    embed_dim : int, optional
        transformer embedding dimension
    depth : int, optional
        depth of attention blocks
    num_heads : int, optional
        number of attention heads
    qkv_bias : bool, optional
        enable bias for qkv projections if True
    mlp_ratio : float, optional,
        ratio of mlp hidden dim to embedding dim
    drop_path_rate : float, optional
        stochastic depth rate in attention blocks
    norm_layer : Normalization layer.
    pos_embed : Literal[None, "learn", "sinusoidal"]
        type of positional embeddings
    embed_layer : nn.Module, optional
        patch embedding layer
    sample_chans : bool, optional
        If true, then randomly sample the input channels in training
    embed_kwargs: dict, optional
        extra parameters for embed_layer
    block_fn : nn.Module, optional
        function to create blocks
    block_kwargs : dict, optional
        extra parameters for block_fn
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        mask_ratio: float,
        sync_mask: bool = True,
        expects_dict: bool = True,
        pretrained: bool = True,
        num_classes: Optional[int] = None,
        global_pool: Literal["avg", "token"] = "avg",
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        qkv_bias: bool = True,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        pos_embed: Literal[None, "learn", "sinusoidal"] = "sinusoidal",
        embed_layer: nn.Module = ContextPatchEmbed,
        sample_chans: bool = True,
        embed_kwargs: dict = {},
        block_fn: nn.Module = Block,
        block_kwargs: dict = {},
    ):
        super(MaskedEncoder, self).__init__()

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            sample_chans=sample_chans,
            **embed_kwargs,
        )  # patchification with channel embeddings
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_heads = num_heads
        self.img_size = img_size

        # positional embeddings
        if pos_embed == "learn":
            # learnable positional embeddings updated during training
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
            )
        elif pos_embed == "sinusoidal":
            # fixed sinusidal positional embeddings
            length = int(self.num_patches**0.5)
            self.pos_embed = build_2d_sinusoidal_pos_embed(
                length=length,
                embed_dim=embed_dim,
                use_class_token=True,
            )

        elif pos_embed is None and block_fn == Block:
            raise ValueError("No form of positional embedding is provided.")

        elif pos_embed is None:
            self.pos_embed = None

        else:
            raise ValueError(
                f"Unknown positional embedding type {pos_embed}. "
                f"Possible options are: 'learn', 'sinusoidal', or None."
            )

        if block_fn == RoPEBlock:
            block_kwargs["num_patches"] = self.num_patches

        # pretrained encoder will be retained for finetuning tasks and it
        # accepts dict inputs containing both images and channel indices
        self.expects_dict = expects_dict

        # patch shuffle and dropout
        assert 0 <= mask_ratio < 1.0, "Mask probability should be in [0, 1)"
        if mask_ratio == 0:
            logging.warning("Mask ratio 0 is being used for model finetuning")
        self.patch_drop = RandomPatchDropout(prob=mask_ratio, sync_mask=sync_mask)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    **block_kwargs,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        # classification head used for finetuning
        self.head = nn.Linear(embed_dim, num_classes) if num_classes else nn.Identity()
        self.global_pool = global_pool

        # customized weights initialization required by MAE
        self.initialize_weights()

        # load weights from pretrained models over ImageNet-1k
        if pretrained:
            lookup_parameters = (embed_dim, depth, num_heads)
            pretrained_model_exists = True

            if lookup_parameters == (768, 12, 12):
                if patch_size != 16:
                    logging.warning(
                        f"""Patch size {patch_size} does not match loaded checkpoint's
                        patch size 16. Pretrained weights should be finetuned to account
                        for this discrepancy"""
                    )
                pretrained_model_name = "vit_base_patch16_224.mae"
            elif lookup_parameters == (1024, 24, 16):
                if patch_size != 16:
                    logging.warning(
                        f"""Patch size {patch_size} does not match loaded checkpoint's
                        patch size 16. Pretrained weights should be finetuned to account
                        for this discrepancy"""
                    )
                pretrained_model_name = "vit_large_patch16_224.mae"
            elif lookup_parameters == (1280, 32, 16):
                if patch_size != 14:
                    logging.warning(
                        f"""Patch size {patch_size} does not match loaded checkpoint's
                        patch size 14. Pretrained weights should be finetuned to account
                        for this discrepancy"""
                    )
                pretrained_model_name = "vit_huge_patch14_224.mae"
            else:
                logging.warning(
                    f"""Current model setting {lookup_parameters} does not match any
                    available pretrained checkpoints in HuggingFace. So pretrained
                    weights are not loaded and randomly initialized weights are used"""
                )
                pretrained_model_exists = False

            if pretrained_model_exists:
                self.load_pretrained_weights(pretrained_model_name)

    def load_pretrained_weights(self, pretrained_model_name: str):
        """Load weights from pretrained models

        In pretrained models, their positional embeddings are fixed sinusoidal
        embeddings. (1) These pretrained embeddings may not align correctly with
        the patch tokens generated with a different patch size. (2) They cannot
        provide learnable or other types of positional embeddings. Thus the
        positional embeddings of pretrained models are not loaded.

        Moreover, pretrained patch embeddings only apply to fixed RGB channels while
        our patch embedding layers use a new channel-agnostic method with 3D Conv layer
        and context embedding layer. So the old patch embeddings weights have
        incompatible shape and should be dropped.

        Therefore, we only load the weights of attention blocks and normalization layers
        that are independent of the patch size and image size.
        """
        pretrained_model = get_named_model(pretrained_model_name, pretrained=True)
        pretrained_state_dict = pretrained_model.state_dict()

        ignored_parameters = [
            "pos_embed",
            "patch_embed.proj.weight",
            "patch_embed.proj.bias",
        ]
        for k in ignored_parameters:
            del pretrained_state_dict[k]

        # only the matching attention blocks and normalization layers will be updated
        self.load_state_dict(pretrained_state_dict, strict=False)

    def initialize_weights(self):
        """Initialize weights and special tokens using trunc_normal"""
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)

        # initialize patch_embed like nn.Linear (instead of nn.Conv3d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view(w.shape[0], -1))

        # recursive initialization for nn.Linear modules that
        # uses xavier_uniform following official JAX ViT
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Weight initialization for linear layers"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def pool(self, x: torch.Tensor, pool_type: str) -> torch.Tensor:
        """Global pooling"""
        if pool_type == "avg":
            x = x[:, 1:].mean(dim=1)  # global pool without cls token
            x = self.norm(x)
        elif pool_type == "token":
            x = x[:, 0]  # use cls token

        return x

    def forward(self, data: Dict[str, torch.Tensor], embed: bool = False):
        """Encoder forward pass"""
        if isinstance(data, dict):
            imgs = data["inputs"]
            # update channel indices for this input batch
            # clone is necessary to avoid overwriting original data
            self.patch_embed.inputs_channel_idx = data["inputs_channel_idx"].clone()
        else:
            raise TypeError("Type %s not valid input to model", type(data))

        # generate patches and add channel embeddings
        # imgs: (B, in_chans, img_size, img_size)
        # -> x: (B, sub_chans * num_patches, embed_dim)
        # where sub_chans is the number of subsampled channels for current batch
        x = self.patch_embed(imgs)

        assert x.shape[1] % self.num_patches == 0
        sub_chans = x.shape[1] // self.num_patches

        # reshape x to (B, sub_chans, num_patches, embed_dim)
        x = x.reshape(x.shape[0], sub_chans, self.num_patches, self.embed_dim)

        # add positional embeddings to x without cls token
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, 1:].unsqueeze(1)

        # random shuffle and patch dropout, output shapes:
        # x: (B, sub_chans, len_keep, embed_dim)
        # where len_keep is the length of kept spatial tokens for each channel
        # ids_resotre: (B, sub_chans, num_patches), used for unshuffling patches
        # mask: (B, sub_chans, num_patches), used for computing loss and visualization
        # ids_keep: (B, sub_chans, len_keep), used to select rope_embed of kept patches
        x, ids_restore, mask, ids_keep = self.patch_drop(x)

        # add cls token
        x = x.reshape(
            x.shape[0], -1, x.shape[-1]
        )  # (B, sub_chans * len_keep, embed_dim)

        if self.pos_embed is not None:
            cls_token = self.cls_token + self.pos_embed[:, :1]  # (1, 1, embed_dim)
        else:
            cls_token = self.cls_token

        cls_tokens = cls_token.expand(x.shape[0], -1, -1)  # (B, 1, embed_dim)
        x = torch.cat(
            (cls_tokens, x), dim=1
        )  # (B, sub_chans * len_keep + 1, embed_dim)

        # attention blocks
        for block in self.blocks:
            if block.__class__.__name__ == "Block":
                x = block(x)
            elif block.__class__.__name__ == "RoPEBlock":
                x = block(x, sub_chans, ids_keep)

        x = self.norm(x)  # (B, sub_chans * len_keep + 1, embed_dim)

        # get image representations
        embeddings = self.head(self.pool(x, self.global_pool))  # (B, embed_dim)

        if embed:
            # image embeddings for featurization or finetuning (with mask_ratio = 0)
            return embeddings

        # channel-subsampled images used as target in loss computation
        # shape of target_imgs: (B, sub_chans, img_size, img_size)
        target_imgs = imgs[:, self.patch_embed.sample_indices]

        return x, ids_restore, mask, sub_chans, target_imgs, embeddings

