from typing import Optional, Tuple

import torch


def build_2d_sinusoidal_pos_embed(
    length: int,
    embed_dim: int,
    use_class_token: bool = True,
    temperature: float = 1e4,
) -> torch.Tensor:
    """
    2D sinusoidal (sine-cosine) position embedding

    This function generates positional embeddings for both the height
    and the width dimensions of the image. In general, sinusoidal pos
    embeddings are fixed and not updated by gradients during training.

    References
    ----------
    https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
    https://github.com/facebookresearch/moco-v3/blob/main/vits.py

    Parameters
    ----------
    length : int
        number of patches along height or width of image after patching
    embed_dim : int
        transformer embedding dimension
    use_class_token : bool
        if add positional embedding to class token
    temperature : float, optional
        scaling factor used in the sinusoidal functions to control the frequency
        of the sine and cosine waves, default is 10000

    Returns
    -------
    pos_emb : torch.Tensor
        positional embeddings for spatial tokens with shape (1, h*w, embed_dim)

    """
    assert (
        embed_dim % 4 == 0
    ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
    grid = torch.arange(length, dtype=torch.float32)
    # create 2D coordinates
    grid_h, grid_w = torch.meshgrid(grid, grid, indexing="ij")
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature**omega)

    # take outer products for each dim
    out_h = torch.outer(grid_h.flatten(), omega)
    out_w = torch.outer(grid_w.flatten(), omega)

    # shape of pos_emb = (1, num_patches, embed_dim)
    pos_emb = torch.cat(
        [torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w)], dim=-1
    ).unsqueeze(0)

    if use_class_token:
        pos_cls = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_emb = torch.cat([pos_cls, pos_emb], dim=1)

    # sin-cos positional embeddings are fixed and not updated in training
    pos_emb = torch.nn.Parameter(pos_emb, requires_grad=False)

    return pos_emb


def init_random_2d_freqs(
    dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True
) -> torch.Tensor:
    """
    Initialize random 2D frequencies for positional embeddings.

    References
    ----------
    https://github.com/naver-ai/rope-vit/blob/main/deit/models_v2_rope.py
    https://github.com/facebookresearch/deit
    https://github.com/meta-llama/codellama/blob/main/llama/model.py

    Parameters
    ----------
    dim : int
        Dimension of the frequencies.
    num_heads : int
        Number of heads.
    theta : float, optional
        Scaling factor for the frequencies, by default 10.0.
    rotate : bool, optional
        Whether to rotate the angles, by default True.

    Returns
    -------
    torch.Tensor
        Initialized random 2D frequencies.
    """
    # freqs_x and freqs_y are empty lists that will store the frequency components for
    # each head
    freqs_x = []
    freqs_y = []

    # mag is a 1D tensor of shape (dim // 4). It represents the magnitudes of the
    # frequencies.
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    for _ in range(num_heads):
        # angles is a 1D tensor of shape (1). It represents the angles of the
        # frequencies.
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)

        # fx and fy are 1D tensors of shape (dim // 2). They represent the x and y
        # components of the frequencies.
        fx = torch.cat(
            [mag * torch.cos(angles), mag * torch.cos(torch.pi / 2 + angles)], dim=-1
        )
        fy = torch.cat(
            [mag * torch.sin(angles), mag * torch.sin(torch.pi / 2 + angles)], dim=-1
        )

        freqs_x.append(fx)
        freqs_y.append(fy)

    # freqs_x and freqs_y are stacked along a new first dimension, resulting in
    # 2D tensors of shape (num_heads, dim // 2).
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)

    # freqs_x and freqs_y are stacked along a new first dimension, resulting in a
    # 3D tensor of shape (2, num_heads, dim // 2).
    freqs = torch.stack([freqs_x, freqs_y], dim=0)

    return freqs


def compute_axial_cis(
    dim: int, end_x: int, end_y: int, theta: float = 10.0
) -> torch.Tensor:
    """
    Compute axial cis frequencies.

    References
    ----------
    https://github.com/naver-ai/rope-vit/blob/main/deit/models_v2_rope.py
    https://github.com/facebookresearch/deit
    https://github.com/meta-llama/codellama/blob/main/llama/model.py

    Parameters
    ----------
    dim : int
        Dimension of the frequencies.
    end_x : int
        End value for x-axis.
    end_y : int
        End value for y-axis.
    theta : float, optional
        Scaling factor for the frequencies, by default 10.0.

    Returns
    -------
    torch.Tensor
        Computed axial cis frequencies.
    """
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)

    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)

    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)

    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def init_t_xy(end_x: int, end_y: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize t_x and t_y values.

    References
    ----------
    https://github.com/naver-ai/rope-vit/blob/main/deit/models_v2_rope.py
    https://github.com/facebookresearch/deit
    https://github.com/meta-llama/codellama/blob/main/llama/model.py

    Parameters
    ----------
    end_x : int
        End value for x-axis.
    end_y : int
        End value for y-axis.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Initialized t_x and t_y values.
    """
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()

    return t_x, t_y


def compute_mixed_cis(
    freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int
) -> torch.Tensor:
    """
    Compute mixed cis frequencies.

    References
    ----------
    https://github.com/naver-ai/rope-vit/blob/main/deit/models_v2_rope.py
    https://github.com/facebookresearch/deit
    https://github.com/meta-llama/codellama/blob/main/llama/model.py

    Parameters
    ----------
    freqs : torch.Tensor
        Frequencies.
    t_x : torch.Tensor
        t_x values.
    t_y : torch.Tensor
        t_y values.
    num_heads : int
        Number of heads.

    Returns
    -------
    torch.Tensor
        Computed mixed cis frequencies.
    """
    t_x = t_x.to(freqs.device)
    t_y = t_y.to(freqs.device)

    N = t_x.shape[0]
    depth = freqs.shape[1]
    freqs_x = (
        (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2))
        .view(depth, N, num_heads, -1)
        .permute(0, 2, 1, 3)
    )
    freqs_y = (
        (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2))
        .view(depth, N, num_heads, -1)
        .permute(0, 2, 1, 3)
    )
    freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)

    return freqs_cis.to(freqs.device)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequencies for broadcasting.

    References
    ----------
    https://github.com/naver-ai/rope-vit/blob/main/deit/models_v2_rope.py
    https://github.com/facebookresearch/deit
    https://github.com/meta-llama/codellama/blob/main/llama/model.py

    Parameters
    ----------
    freqs_cis : torch.Tensor
        Frequencies.
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Reshaped frequencies.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-4], x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 4 else 1 for i, d in enumerate(x.shape)]
    else:
        raise ValueError(
            f"freqs_cis shape {freqs_cis.shape} does not match input shape {x.shape}"
        )

    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings.

    References
    ----------
    https://github.com/naver-ai/rope-vit/blob/main/deit/models_v2_rope.py
    https://github.com/facebookresearch/deit
    https://github.com/meta-llama/codellama/blob/main/llama/model.py

    Parameters
    ----------
    xq : torch.Tensor
        Query tensor.
    xk : torch.Tensor
        Key tensor.
    freqs_cis : torch.Tensor
        Frequencies.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Output query tensor and key tensor.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


def gather_freqs_cis(
    freqs_cis: torch.Tensor, ids_keep: Optional[torch.Tensor], sub_chans: int
) -> torch.Tensor:
    """
    Gathers the tensor freqs_cis based on the ids_keep and subchannels.

    Parameters
    ----------
    freqs_cis : torch.Tensor
        The input tensor with shape [B, H, num_patches, head_dim // 2].
    ids_keep : Optional[torch.Tensor]
        The indices to gather with shape [B, sub_chans, len_keep].
    sub_chans : int
        The number of subchannels.

    Returns
    -------
    torch.Tensor
        The reduced tensor with shape [B, H, sub_chans * len_keep, head_dim //2].
    """
    _, H, _, embed_dim = freqs_cis.shape  # here embed_dim = head_dim // 2

    if ids_keep is not None:
        # Expand ids_keep to add head dimensions of freqs_cis
        ids_keep_expanded = (
            ids_keep.flatten(1).unsqueeze(1).expand(-1, H, -1)
        )  # [B, H, sub_chans * len_keep]

        # For each head, all subchannels share the same frequency in that head. So we
        # can gather along the sequence length dimension (dim=2) for all subchannels.
        index = ids_keep_expanded.unsqueeze(-1).repeat(1, 1, 1, embed_dim)

        freqs_cis_gathered = torch.gather(
            freqs_cis, dim=2, index=index
        )  # [B, H, sub_chans * len_keep, embed_dim]

        return freqs_cis_gathered

    # keep all patches if ids_keep is None
    return freqs_cis.repeat(
        1, 1, sub_chans, 1
    )  # [B, H, sub_chans * len_keep, embed_dim]
