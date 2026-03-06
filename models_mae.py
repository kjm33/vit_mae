# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
#
# Masked Autoencoder (MAE) for self-supervised vision learning.
# The model masks a large fraction of image patches, encodes only the visible
# patches with a ViT encoder, then reconstructs the full image with a lightweight
# decoder. Reconstruction loss is computed only on masked patches.

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """
    Masked Autoencoder with Vision Transformer (ViT) backbone.

    High-level flow:
      1. Split image into patches and embed them.
      2. Randomly mask most patches (e.g. 75%); keep only a subset visible.
      3. Encode visible patches with a ViT encoder (no masked tokens in encoder).
      4. Decoder gets full sequence: re-insert mask tokens, unshuffle to original order.
      5. Decoder predicts pixel values per patch; loss is MSE on masked patches only.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --- Input / grid setup ---
        self.in_chans = in_chans
        self.patch_size = patch_size if isinstance(patch_size, int) else patch_size[0]
        # Number of patches along each spatial dimension (H, W in patch units)
        if isinstance(img_size, (list, tuple)):
            self._grid_size = (img_size[0] // self.patch_size, img_size[1] // self.patch_size)
        else:
            g = img_size // self.patch_size
            self._grid_size = (g, g)

        # --------------------------------------------------------------------------
        # MAE encoder: processes only the *visible* (non-masked) patches
        # --------------------------------------------------------------------------
        # Linear projection of each patch (typically via Conv2d) into embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches  # total patches, e.g. 14*14 for 224/16

        # [CLS] token (one per sequence); gets same treatment as in BERT/ViT
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional embedding for all positions (1 + num_patches for cls + patches). Fixed, not learned.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        # Stack of Transformer blocks (self-attention + MLP)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder: reconstructs full image from encoder output + mask tokens
        # --------------------------------------------------------------------------
        # Project encoder output (embed_dim) down to decoder dimension (often smaller)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # Learnable token that replaces each masked patch in the decoder input
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Decoder positional embedding (full sequence length; fixed sin-cos)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        # Lightweight Transformer stack for the decoder
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # Final projection: each decoder token -> patch_size^2 * in_chans (pixel values of one patch)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
        # --------------------------------------------------------------------------

        # If True, normalize patch pixels (zero mean, unit var) before computing loss
        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize encoder/decoder weights. Position embeddings use fixed 2D sin-cos; rest use standard init."""
        grid_size = self._grid_size

        # Encoder: fill pos_embed with 2D sin-cos (no learnable params; requires_grad=False keeps it fixed)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Decoder: same 2D sin-cos for decoder positions
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], grid_size, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Patch embedding projection: Xavier init as in original ViT (treat conv kernel as linear weight matrix)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Learnable tokens: small random init (std=0.02)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # All Linear and LayerNorm layers in encoder/decoder
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Per-module init: Xavier for Linear, standard for LayerNorm."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        Flatten image into a sequence of patch vectors (same layout as patch_embed output).
        Used to get target patch values for the reconstruction loss.

        Args:
            imgs: (N, C, H, W) — batch of images
        Returns:
            x: (N, L, patch_size**2 * C) — L = h*w patches, each patch is p*p*C values
        """
        p = self.patch_embed.patch_size[0]
        c = self.in_chans
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p  # number of patches along height
        w = imgs.shape[3] // p  # number of patches along width
        # Reshape so we can extract p x p blocks: (N, C, h, p, w, p) then reorder to (N, h, w, p, p, C)
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)  # group pixels by patch
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))  # (N, L, patch_vec_dim)
        return x

    def unpatchify(self, x):
        """
        Convert patch sequence back to image shape. Inverse of patchify; used for visualization.

        Args:
            x: (N, L, patch_size**2 * C) — decoder output (one vector per patch)
        Returns:
            imgs: (N, C, H, W) — reconstructed images
        """
        p = self.patch_embed.patch_size[0]
        c = self.in_chans
        h, w = self._grid_size
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)  # (N, C, h, p, w, p)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Randomly mask a fraction of patches per sample. Uses shuffling (no padding):
        we sort patches by random noise and keep the first (1 - mask_ratio)*L.

        Args:
            x: (N, L, D) — patch sequence (no cls token yet)
            mask_ratio: fraction of patches to mask (e.g. 0.75 -> keep 25%)
        Returns:
            x_masked: (N, len_keep, D) — only the kept (visible) patches
            mask: (N, L) — 0 = kept, 1 = masked (in original patch order)
            ids_restore: (N, L) — indices to unshuffle x_masked back to full order (for decoder)
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        # Random permutation per sample: argsort(noise) gives random order; first len_keep = "keep"
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)   # ascending: small index = keep
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # inverse: restore original order

        # Keep only the first len_keep positions (after shuffle)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Binary mask in *original* patch order: 0 = keep, 1 = remove (for loss weighting)
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """
        Encode only the visible (non-masked) patches. No mask tokens in the encoder.

        Steps: patch embed -> add position (patches only) -> random mask -> append cls -> Transformer -> norm.
        """
        # (N, C, H, W) -> (N, L, embed_dim)
        x = self.patch_embed(x)

        # Add positional embedding for patches only (pos_embed[:, 1:, :] skips the cls position)
        x = x + self.pos_embed[:, 1:, :]

        # Random masking: (N, L, D) -> (N, len_keep, D); also get mask and ids_restore for decoder
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # Prepend [CLS] token (with its position) so sequence is [cls, patch_1, ..., patch_len_keep]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # ViT encoder blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        Decode full sequence and predict patch pixels. Reconstructs full patch order
        by re-inserting mask tokens and unshuffling with ids_restore.

        Input x: (N, 1 + len_keep, embed_dim) — [cls, visible patches]
        Output: (N, L, patch_size**2 * C) — one vector per patch (no cls)
        """
        # Project encoder dim -> decoder dim
        x = self.decoder_embed(x)

        # Restore full sequence: x has [cls, len_keep patches]. We need [cls, L patches].
        # x[:, 1:, :] = visible patches; we need (L - len_keep) mask tokens to fill the gaps.
        num_mask = ids_restore.shape[1] + 1 - x.shape[1]  # L - len_keep ( +1 for cls already in x)
        mask_tokens = self.mask_token.repeat(x.shape[0], num_mask, 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)   # [visible | mask] in *shuffled* order
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle to original order
        x = torch.cat([x[:, :1, :], x_], dim=1)  # [cls | patch_1 ... patch_L]

        # Add decoder positional embedding (full length)
        x = x + self.decoder_pos_embed

        # Decoder Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predict pixel values per patch: (N, 1+L, decoder_dim) -> (N, 1+L, p*p*C)
        x = self.decoder_pred(x)

        # Drop cls token for loss: (N, L, p*p*C)
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        MSE reconstruction loss computed *only on masked patches* (mask=1).
        Optionally normalize patch pixels (norm_pix_loss) for more stable training.

        Args:
            imgs: (N, C, H, W) — original images
            pred: (N, L, p*p*C) — decoder predictions per patch
            mask: (N, L) — 0 = visible (ignore), 1 = masked (predict these)
        Returns:
            Scalar: mean squared error over masked patches only
        """
        target = self.patchify(imgs)  # (N, L, p*p*C) — ground-truth patch vectors
        if self.norm_pix_loss:
            # Normalize each patch to zero mean, unit variance (per patch, per channel)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (N, L) — MSE per patch

        # Only backprop on masked positions: (loss * mask).sum() / mask.sum()
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        """
        Full MAE forward: encode (with masking), decode, compute reconstruction loss.

        Args:
            imgs: (N, C, H, W)
            mask_ratio: fraction of patches to mask (default 0.75)
        Returns:
            loss: scalar reconstruction loss (on masked patches)
            pred: (N, L, p*p*C) — predicted patch values (for visualization / metrics)
            mask: (N, L) — binary mask (1 = masked)
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


# ---------------------------------------------------------------------------
# Model factory functions: standard MAE architectures (encoder varies, decoder fixed)
# ---------------------------------------------------------------------------
# Naming: mae_vit_{base|large|huge}_patch{P}_dec512d8b = ViT-{B/L/H}, 16x16 or 14x14 patches, decoder 512-dim 8 blocks


def mae_vit_base_patch16_dec512d8b(**kwargs):
    """MAE with ViT-Base: 768-dim, 12 layers, 12 heads; decoder 512-dim, 8 blocks."""
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    """MAE with ViT-Large: 1024-dim, 24 layers, 16 heads; decoder 512-dim, 8 blocks."""
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    """MAE with ViT-Huge: 1280-dim, 32 layers, 16 heads, 14x14 patches; decoder 512-dim, 8 blocks."""
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch8_32x512_dec512d8b(**kwargs):
    """MAE for text/grayscale: img 32x512, patch 8, 1 channel, ViT-Base; decoder 512-dim, 8 blocks. norm_pix_loss=True."""
    model = MaskedAutoencoderViT(
        img_size=(32, 512),
        patch_size=8,
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_pix_loss=True,
        **kwargs)
    return model


# Recommended aliases (same configs, shorter names)
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b
mae_vit_base_patch8_32x512 = mae_vit_base_patch8_32x512_dec512d8b
