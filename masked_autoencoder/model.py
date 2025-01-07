import logging
from functools import partial
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from channel_agnostic_vit.masked_autoencoder.modules import (
    ContextPatchEmbed,
    RandomPatchDropout,
    RoPEBlock,
)
from channel_agnostic_vit.masked_autoencoder.position_embedding import build_2d_sinusoidal_pos_embed



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


class MaskedDecoder(nn.Module):
    """
    Decoder part of masked autoencoder

    The default setting uses a light ViT encoder with
    (patch_size, embed_dim, depth, num_heads) = (16, 512, 8, 16)

    Parameters
    ----------
    num_patches : int
        number of patches in input images
    patch_size : int
        patch size
    encoder_embed_dim : int, optional
        embedding dimension of encoder
    embed_dim : int, optional
        embedding dimension of decoder
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
    block_fn : nn.Module, optional
        function to create blocks
    block_kwargs : dict, optional
        extra parameters for block_fn
    """

    def __init__(
        self,
        num_patches: int,
        patch_size: int,
        encoder_embed_dim: int,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 16,
        qkv_bias: bool = True,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        pos_embed: Literal[None, "learn", "sinusoidal"] = "sinusoidal",
        block_fn: nn.Module = Block,
        block_kwargs: dict = {},
    ):
        super(MaskedDecoder, self).__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.num_heads = num_heads

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

        # linear projection for channel embeddings
        self.context_layer = nn.Linear(encoder_embed_dim, embed_dim)

        self.decoder_embed = nn.Linear(encoder_embed_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
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
        self.pred = nn.Linear(embed_dim, patch_size**2)

        # customized weights initialization required by MAE
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights and special tokens using trunc_normal"""
        torch.nn.init.trunc_normal_(self.mask_token, std=0.02)

        # recursive initialization for nn.Linear modules that
        # uses xavier_uniform following official JAX ViT
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Weight initialization for linear layers"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def unpatchify(self, x: torch.Tensor):
        """
        Unpatchify patch tokens to images

        input x: (B, C, num_patches, patch_size**2)
        output imgs: (B, C, img_size, img_size)
        """
        B, C = x.shape[0], x.shape[1]
        h = w = int(self.num_patches**0.5)
        x = x.reshape(B, C, h, w, self.patch_size, self.patch_size)
        imgs = x.permute(0, 1, 2, 4, 3, 5).reshape(
            B, C, h * self.patch_size, w * self.patch_size
        )
        return imgs

    def forward(
        self,
        x: torch.Tensor,
        channel_embeddings: torch.Tensor,
        ids_restore: torch.Tensor,
        mask: torch.Tensor,
        sub_chans: int,
    ):
        """Decoder forward pass"""
        x = self.decoder_embed(x)  # (B, sub_chans * len_keep + 1, embed_dim)
        assert (x.shape[1] - 1) % sub_chans == 0
        B, len_keep = x.shape[0], (x.shape[1] - 1) // sub_chans

        if self.training:
            # append mask tokens to sequence
            mask_tokens = self.mask_token.unsqueeze(1).repeat(
                B, sub_chans, self.num_patches - len_keep, 1
            )  # (B, sub_chans, num_patches - len_keep, embed_dim)
            x_ = torch.cat(
                (x[:, 1:].reshape(B, sub_chans, len_keep, -1), mask_tokens), dim=2
            )  # no cls token

            # unshuffle tokens to restore the original patch order
            ids_restore = ids_restore.unsqueeze(-1)  # (B, sub_chans, num_patches, 1)
            ids_restore = ids_restore.repeat(
                1, 1, 1, x.shape[-1]
            )  # (B, sub_chans, num_patches, embed_dim)
            x_ = torch.gather(
                x_,
                dim=2,
                index=ids_restore,
            )  # (B, sub_chans, num_patches, embed_dim)
        else:
            # no need to append and unshuffle in testing
            assert len_keep == self.num_patches
            x_ = x[:, 1:].reshape(B, sub_chans, len_keep, -1)

        # add channel embed
        context_embeddings = self.context_layer(channel_embeddings).unsqueeze(
            2
        )  # (B, sub_chans, 1, embed_dim)
        x_ = x_ + context_embeddings  # (B, sub_chans, num_patches, embed_dim)

        # add pos embed
        if self.pos_embed is not None:
            x_ = x_ + self.pos_embed[:, 1:].unsqueeze(1)
        x_ = x_.reshape(B, -1, x_.shape[-1])

        # append cls token
        if self.pos_embed is not None:
            x = torch.cat((x[:, :1] + self.pos_embed[:, :1], x_), dim=1)
        else:
            x = torch.cat((x[:, :1], x_), dim=1)

        # attention blocks
        for block in self.blocks:
            if block.__class__.__name__ == "Block":
                x = block(x)
            elif block.__class__.__name__ == "RoPEBlock":
                x = block(x, sub_chans)

        x = self.norm(x)  # (B, sub_chans * num_patches + 1, embed_dim)

        # predictor projection
        x = self.pred(x)  # (B, sub_chans * num_patches + 1, patch_size**2)

        # remove cls token and reshape the predicted tokens to reconstructed images
        x = x[:, 1:, :]  # (B, sub_chans * num_patches, patch_size**2)
        x = x.reshape(B, sub_chans, self.num_patches, -1)
        pred_imgs = self.unpatchify(x)  # (B, sub_chans, img_size, img_size)

        # reshape mask (B, sub_chans, num_patches) -> (B, sub_chans, img_size, img_size)
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, self.patch_size**2)
        mask_imgs = self.unpatchify(mask)

        return pred_imgs, mask_imgs


class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder using ViT

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
    encoder_embed_dim : int, optional
        embedding dimension of encoder
    encoder_depth : int, optional
        depth of encoder attention blocks
    encoder_num_heads : int, optional
        number of encoder attention heads
    decoder_embed_dim : int, optional
        embedding dimension of decoder
    decoder_depth : int, optional
        depth of decoder attention blocks
    decoder_num_heads : int, optional
        number of decoder attention heads
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

    References
    ----------
    Masked Autoencoders Are Scalable Vision Learners (https://arxiv.org/abs/2111.06377)
    timm from Hugging Face (https://huggingface.co/docs/timm/index)
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        mask_ratio: float,
        sync_mask: bool = True,
        expects_dict: bool = True,
        num_classes: Optional[int] = None,
        global_pool: Literal["avg", "token"] = "avg",
        encoder_embed_dim: int = 1024,
        encoder_depth: int = 24,
        encoder_num_heads: int = 16,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
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
        super(MaskedAutoencoder, self).__init__()
        self.expects_dict = expects_dict  # if receive dict inputs

        self.encoder = MaskedEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            mask_ratio=mask_ratio,
            sync_mask=sync_mask,
            expects_dict=expects_dict,
            num_classes=num_classes,
            global_pool=global_pool,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            qkv_bias=qkv_bias,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            pos_embed=pos_embed,
            embed_layer=embed_layer,
            sample_chans=sample_chans,
            embed_kwargs=embed_kwargs,
            block_fn=block_fn,
            block_kwargs=block_kwargs,
        )

        self.decoder = MaskedDecoder(
            num_patches=self.encoder.num_patches,
            patch_size=patch_size,
            encoder_embed_dim=encoder_embed_dim,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            qkv_bias=qkv_bias,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            pos_embed=pos_embed,
            block_fn=block_fn,
            block_kwargs=block_kwargs,
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass"""
        latent, ids_restore, mask, sub_chans, target_imgs, embeddings = self.encoder(
            data, embed=False
        )

        # get channel embeddings from encoder that doesn't require gradient,
        # i.e. channel_embeddings is a leaf node in the computational graph
        channel_embeddings = self.encoder.patch_embed.channel_embeddings

        pred_imgs, mask_imgs = self.decoder(
            latent, channel_embeddings, ids_restore, mask, sub_chans
        )
        out = {
            "embeddings": embeddings,
            "pred_imgs": pred_imgs,
            "target_imgs": target_imgs,
            "mask_imgs": mask_imgs,
        }
        return out
