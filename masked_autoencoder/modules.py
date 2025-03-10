from typing import Literal, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from timm.layers import Mlp
from timm.models.vision_transformer import Attention, Block
from torch import nn

from masked_autoencoder.position_embedding import (
    apply_rotary_emb,
    compute_axial_cis,
    compute_mixed_cis,
    gather_freqs_cis,
    init_random_2d_freqs,
    init_t_xy,
)


class ContextPatchEmbed(nn.Module):
    """
    Context Patch Embed

    Takes image and creates a sequence of embeddings,
    with each embedding corresponding to a patch of
    the input image.

    This is done for each channel of the image
    independently. To provide model with
    information about which channel the patch
    is from, the embedding is modified by a
    'channel embedding'. In this child class,
    the channel embedding is the average embedding
    of all the patches from that channel in the batch,
    projected through a single dense layer.

    This allows channel embeddings to be learned by the
    model, whilst being agnostic to the channels that are
    seen during training and inference when computing
    channel embeddings.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        sample_chans: bool = True,
        chan_embed_mode: Literal["sample", "batch", "global"] = "global",
        encoder_chan_embed: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv3d(
            1,
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )

        # use channel embeddings in the encoder of MAE
        if encoder_chan_embed:
            self.context_layer = nn.Linear(embed_dim, embed_dim)

        self.sample_chans = sample_chans

        # specify the way to average channel embeddings
        if chan_embed_mode == "sample":
            self.batch_avg = False
            self.global_batch = False
        elif chan_embed_mode == "batch":
            self.batch_avg = True
            self.global_batch = False
        elif chan_embed_mode == "global":
            self.batch_avg = True
            self.global_batch = True

        self.encoder_chan_embed = encoder_chan_embed

    @torch.no_grad()
    def average_channel_embedding(self, x: torch.Tensor):
        """
        Compute Average Channel Embedding without Gradients

        Given a particular channel index, find all embeddings in the
        global batch from that channel and compute the mean.

        Creates a tensor which contains,
        the channel embedding for each patch
        embedding, where patches from the same,
        channel share the same channel embedding.

        Arguments
        ---------
        x: patchified tensor with shape (B, embed_dim, C, H, W)

        Returns
        -------
        av_channel_embed: tensor containing channel embedding for each patch embedding
        """
        av_patch_embed = torch.mean(x, dim=(3, 4)).transpose(
            0, 1
        )  # shape = (embed_dim, B, C)

        if not self.batch_avg:
            return av_patch_embed

        if self.global_batch and dist.is_available() and dist.is_initialized():
            # compute channel average across all distributed processes, i.e. GPUs
            # using dist.all_gather function that expects contiguous tensors
            av_patch_embed = av_patch_embed.contiguous()
            B = av_patch_embed.shape[1]  # batch size
            world_size = dist.get_world_size()
            gathered_inputs_channel_idx = [
                torch.zeros_like(self.inputs_channel_idx) for _ in range(world_size)
            ]
            dist.all_gather(gathered_inputs_channel_idx, self.inputs_channel_idx)
            gathered_inputs_channel_idx = torch.cat(gathered_inputs_channel_idx, dim=0)
            # shape of gathered_inputs_channel_idx = (world_size * B, C)
            gathered_av_patch_embed = [
                torch.zeros_like(av_patch_embed) for _ in range(world_size)
            ]
            dist.all_gather(gathered_av_patch_embed, av_patch_embed)
            gathered_av_patch_embed = torch.cat(gathered_av_patch_embed, dim=1)
            # shape of gathered_av_patch_embed = (embed_dim, world_size * B, C)

            gathered_av_channel_embed = torch.zeros_like(gathered_av_patch_embed)
            for idx in gathered_inputs_channel_idx.unique():
                # get all embeddings where C = input_channel
                where_channel = torch.where(
                    gathered_inputs_channel_idx == idx, True, False
                )
                channel_embed = gathered_av_patch_embed[
                    :, where_channel
                ]  # channel_embed.shape = (embed_dim, num. where channel is idx)
                channel_mean = channel_embed.mean(dim=1, keepdim=True)
                gathered_av_channel_embed[:, where_channel] = channel_mean

            # get average channel embeddings for current process
            rank = dist.get_rank()
            av_channel_embed = gathered_av_channel_embed[
                :, B * rank : B * (rank + 1)
            ]  # (embed_dim, B, C)
        else:
            # compute channel average within the current batch when
            # DDP is not used
            av_channel_embed = torch.zeros_like(av_patch_embed)

            for idx in self.inputs_channel_idx.unique():
                # get all embeddings where C = input_channel
                where_channel = torch.where(self.inputs_channel_idx == idx, True, False)
                channel_embed = av_patch_embed[
                    :, where_channel
                ]  # channel_embed.shape = (embed_dim, num. where channel is idx)
                channel_embed = torch.mean(
                    channel_embed, dim=1, keepdim=True
                )  # shape = (embed_dim, 1)
                av_channel_embed[:, where_channel] = channel_embed  # (embed_dim, B, C)

        return av_channel_embed

    def forward(self, x: torch.Tensor):
        """
        Forward Pass

        - Project input into patches via Conv3d projection layer
        - Compute average embedding across each channel for each sample in batch
        - Find all embeddings for each channel across all samples in batch, and
        compute the average embedding representing each channel in batch
        - Detach the channel embeddings, and project them through a dense layer
          to yield context embeddings
        - Add these back to the input to encode channel information via
          context embeddings
        """
        if self.training and self.sample_chans:
            # sample input channels to boost inference performance on sparse channels
            C = np.random.randint(1, x.shape[1] + 1)
            sample_indices = np.random.choice(x.shape[1], size=C, replace=False)
            x = x[:, sample_indices]
            self.inputs_channel_idx = self.inputs_channel_idx[:, sample_indices]
            self.sample_indices = sample_indices
        else:
            self.sample_indices = np.arange(x.shape[1])

        # shared projection layer across channels
        x = self.proj(x.unsqueeze(1))  # output shape = (B, embed_dim, C, H, W)

        # compute averge channel embedding without gradients
        av_channel_embed = self.average_channel_embedding(x)  # (embed_dim, B, C)
        av_channel_embed = av_channel_embed.permute(1, 2, 0)  # (B,C,embed_dim)
        self.channel_embeddings = av_channel_embed  # used by the decoder of MAE

        if self.encoder_chan_embed:
            # get context channel embedding that requires gradients
            context_embeddings = self.context_layer(av_channel_embed)
            context_embeddings = context_embeddings.permute(0, 2, 1)
            context_embeddings = context_embeddings.unsqueeze(-1).unsqueeze(-1)

            # add channel embedding to corresponding channels
            x += context_embeddings  # shape = (B, embed_dim, C, H, W)

        x = x.flatten(2)  # shape = (B, embed_dim, CHW)
        x = x.transpose(1, 2)  # shape = (B, CHW, embed_dim)

        return x


class RandomPatchDropout(nn.Module):
    """
    Patch dropout after per-sample random shuffle

    Perform per-sample random shuffle and drop masked patch tokens
    where per-sample shuffling is done by argsort random noise.

    Parameters
    ----------
    prob : float
        ratio that patches are randomly dropped
    sync_mask : bool
        if true, the same random patche dropout is done across all channels.
        Otherwise, random patch dropout is performed independently for each channel.
    """

    def __init__(
        self,
        prob: float,
        sync_mask: True,
    ):
        super().__init__()
        self.prob = prob
        self.sync_mask = sync_mask

    def forward(self, x):
        """Forward pass"""
        B, C, L, D = x.shape
        # no random shuffle at testing time or prob is 0
        if not self.training or self.prob == 0.0:
            ids_restore = None
            ids_keep = None
            # all-one mask means compute reconstruction loss over all pixels
            mask = torch.ones([B, C, L], device=x.device)
            return x, ids_restore, mask, ids_keep

        # per-sample random shuffle using sorted noise
        # shape of ids_shuffle = (B, C, L)
        if self.sync_mask:
            random_noise = torch.rand(B, L, device=x.device)
            ids_shuffle = (
                torch.argsort(random_noise, dim=1).unsqueeze(1).repeat(1, C, 1)
            )
        else:
            random_noise = torch.rand(B, C, L, device=x.device)
            ids_shuffle = torch.argsort(random_noise, dim=-1)

        ids_restore = torch.argsort(ids_shuffle, dim=-1)  # restore indices
        len_keep = max(1, int(L * (1 - self.prob)))
        ids_keep = ids_shuffle[..., :len_keep]  # (B, C, len_keep)

        # patch dropout: shape of x = (B, C, len_keep, D)
        x = torch.gather(x, dim=2, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))

        # generate the binary mask used in loss computation and visualization
        mask = torch.ones([B, C, L], device=x.device)
        mask[..., :len_keep] = 0
        # unshuffle to get the binary mask, shape of mask = (B, C, L)
        # where 0 means kept patch, 1 means masked (removed) patch
        mask = torch.gather(mask, dim=-1, index=ids_restore)

        return x, ids_restore, mask, ids_keep


class RoPEAttention(Attention):
    """Multi-head Attention block with relative position embeddings.

    References
    ----------
    https://github.com/naver-ai/rope-vit/blob/main/deit/models_v2_rope.py

    """

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RoPEAttention block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        freqs_cis : torch.Tensor
            Frequencies.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)

        # RoPE can only be applied to the non-cls tokens
        q_new, k_new = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)

        q = torch.cat((q[:, :, :1], q_new), dim=2)
        k = torch.cat((k[:, :, :1], k_new), dim=2)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class RoPEBlock(Block):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        rope_mode: str,
        num_patches: int,
        rope_theta: float = 10.0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        """
        Ropeblock constructor.

        Parameters
        ----------
        dim : int
            Dimension of the input tensor.
        num_heads : int
            Number of attention heads.
        rope_mode : str
            Mode parameter for the rope mechanism.
        num_patches : int
            Number of patches in the input.
        rope_theta : float
            Theta parameter for the rope mechanism.
        mlp_ratio : float, optional
            Ratio of MLP hidden dimension to input dimension, by default 4.0.
        qkv_bias : bool, optional
            Whether to include bias in the query, key, and value projections,
            by default False.
        qk_norm : bool, optional
            Whether to normalize the query and key projections, by default False.
        proj_drop : float, optional
            Dropout rate applied to the output of the projection layers, by default 0.0.
        attn_drop : float, optional
            Dropout rate applied to the attention weights, by default 0.0.
        init_values : float, optional
            Initial value for the weights, by default None.
        drop_path : float, optional
            Dropout rate applied to the output of the attention and MLP layers,
            by default 0.0.
        act_layer : nn.Module, optional
            Activation function applied to the output of the attention and MLP layers,
            by default nn.GELU.
        norm_layer : nn.Module, optional
            Normalization layer applied to the output of the attention and MLP layers,
            by default nn.LayerNorm.
        mlp_layer : nn.Module, optional
            MLP layer used in the block, by default Mlp.
        """
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
        )
        self.attn = RoPEAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.num_heads = num_heads
        self.rope_mode = rope_mode
        self.head_dim = dim // num_heads
        self.theta = rope_theta
        num_patches = num_patches
        self.end_x, self.end_y = int(num_patches**0.5), int(num_patches**0.5)

        if self.rope_mode == "mixed":
            # Initialize the frequencies
            freqs = init_random_2d_freqs(
                dim=self.head_dim,
                num_heads=self.num_heads,
                theta=self.theta,
            )
            freqs = freqs.reshape(2, 1, -1)  # (2, depth, H * head_dim // 2)

            self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)

            # t_x t_y initialization
            self.t_x, self.t_y = init_t_xy(
                end_x=int(num_patches**0.5),
                end_y=int(num_patches**0.5),
            )  # (num_patches, num_patches)

        elif self.rope_mode == "axial":
            # fixed frequencies (for each head) that is not learnable
            self.freqs_cis = compute_axial_cis(
                dim=self.head_dim,
                end_x=self.end_x,
                end_y=self.end_y,
                theta=self.theta,
            )  # (num_patches, head_dim // 2)

        else:
            raise ValueError(
                f"Unknown rope embedding type {self.rope_mode}. "
                f"Possible options are: 'mixed', 'axial'."
            )

    def forward(
        self, x: torch.Tensor, sub_chans: int, ids_keep: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the RoPEBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        sub_chans : int
            Number of sub-channels.
        ids_keep : Optional[torch.Tensor]
            Indices of the patches to keep after dropout.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        freqs_cis = self._get_freqs_cis(x, sub_chans, ids_keep)
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), freqs_cis=freqs_cis)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x

    def _get_freqs_cis(self, x, sub_chans, ids_keep):
        if self.rope_mode == "mixed":
            freqs_cis = compute_mixed_cis(
                self.freqs, self.t_x, self.t_y, self.num_heads
            )  # (depth, H, num_patches, head_dim // 2)

            freqs_cis = freqs_cis[0]  # (H, num_patches, head_dim // 2)
            freqs_cis = freqs_cis.unsqueeze(0).repeat(
                x.shape[0], 1, 1, 1
            )  # (B, H, num_patches, head_dim // 2)

        elif self.rope_mode == "axial":
            freqs_cis = (
                self.freqs_cis.unsqueeze(0)
                .unsqueeze(0)
                .repeat(x.shape[0], self.num_heads, 1, 1)
            )  # (B, H, num_patches, head_dim // 2)

        freqs_cis = gather_freqs_cis(
            freqs_cis=freqs_cis.to(x), ids_keep=ids_keep, sub_chans=sub_chans
        )  # [B, H, sub_chans * len_keep, head_dim // 2]

        return freqs_cis.to(x)
