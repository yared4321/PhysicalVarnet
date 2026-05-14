"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import matplotlib.pyplot as plt
import torch
import numpy as np
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import Unet
import fastmri_custom as fastmri # Use local directory as an alias
import fastmri_custom.fftc as fftc # Important for fft2c/ifft2c
from fastmri_custom.data import transforms

# Global values for physical scale (per LUT and metadata)
MAX_TR = 10000.0
MAX_TE = 200.0
MAX_ALPHA = 180.0

# Brainweb values for map scaling (T1, T2, PD)
T1_MAX = 3000.0 
T2_MAX = 500.0
PD_MAX = 1.0

class NormUnet(nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        with_physics: bool = False
        
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()
        self.with_physics = with_physics
        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
            with_physics= with_physics,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w).contiguous()

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()
    
    def norm_5d(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x shape: [B, C, H, W, 2]
        Normalizes real and imaginary separately across coils and pixels.
        """
        if len(x.shape) == 5:
            calc_dims = (1, 2, 3) 
        elif len(x.shape) < 5:
            calc_dims = tuple(range(2, x.ndim))

        mean = x.mean(dim=calc_dims, keepdim=True)
        std = x.std(dim=calc_dims, keepdim=True)
        
        x_norm = (x - mean) / (std + 1e-8)
        
        return x_norm, mean, std
    
    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std
    
    def unnorm(self, x, mean, std):
        return x * std + mean
    
    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        h, w = x.shape[-2], x.shape[-1]
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, (w_pad[0], w_pad[1], h_pad[0], h_pad[1]))
        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        new_x = x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]].contiguous()
        return new_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)
        
        x, physics_params = self.unet(x)
        
        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)

        if not self.with_physics:
            # For cascades 1-11: unnormalize to complex scale
            x = self.unnorm(x, mean, std)
            x = self.chan_complex_to_last_dim(x)
        
        return x, physics_params


class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        mask_center: bool = True,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()
        self.mask_center = mask_center
        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
            with_physics=False
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape
        x = x.view(b * c, 1, h, w, comp)
        return x, b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size
        x = x.view(batch_size, c, h, w, comp)
        return x

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        x= x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)
        # print(f' [DEBUG] [SENSITIVITY] [divide_root_sum_of_squares] x shape: {x.shape}')
        return x 

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        is_empty = (num_low_frequencies is None) or \
                   (torch.is_tensor(num_low_frequencies) and num_low_frequencies.float().mean() == 0) or \
                   (not torch.is_tensor(num_low_frequencies) and num_low_frequencies == 0)

        if is_empty:
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )
        else:
            if torch.is_tensor(num_low_frequencies):
                num_low_frequencies_tensor = num_low_frequencies.to(mask).flatten().long()
            else:
                num_low_frequencies_tensor = torch.full(
                    (mask.shape[0],), 
                    fill_value=int(num_low_frequencies), 
                    dtype=torch.long, 
                    device=mask.device
                )
        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2
        return pad.type(torch.long), num_low_frequencies_tensor.type(torch.long)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        if self.mask_center:
            pad, num_low_freqs = self.get_pad_and_num_low_freqs(
                mask, num_low_frequencies
            )
            masked_kspace = transforms.batched_mask_center(
                masked_kspace, pad, pad + num_low_freqs
            )
        
        # convert to image space
        images, batches = self.chans_to_batch_dim(fftc.ifft2c_new(masked_kspace))
        norm_unet_out, _ = self.norm_unet(images)

        # estimate sensitivities
        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(norm_unet_out, batches)
        )


class VarNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 4,
        sens_pools: int = 3,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()
              
        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )

        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools,out_chans=2,with_physics=False),with_tissue=False) for _ in range(num_cascades-1)]
        )
        self.cascades.append(
            VarNetBlock(NormUnet(chans, pools, out_chans=3, with_physics=True))
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        
        masked_kspace_norm , masked_kspace_mean, masked_kspace_std = self.cascades[-2].model.norm_5d(masked_kspace)
        kspace_pred_norm, kspace_pred_mean, kspace_pred_std  = self.cascades[-2].model.norm_5d(masked_kspace)
        kspace_pred_norm = kspace_pred_norm.clone()

        for i, cascade in enumerate(self.cascades):
            if i == len(self.cascades) - 1:
                kspace_for_physics = kspace_pred_norm.detach()
                tissue_maps, last_p = cascade(kspace_for_physics, masked_kspace_norm, mask, sens_maps)
            else:
                kspace_pred_norm = cascade(kspace_pred_norm, masked_kspace_norm, mask, sens_maps)

        kspace_pred_final = self.cascades[-2].model.unnorm(kspace_pred_norm,kspace_pred_mean, kspace_pred_std)
        image_prior_raw = cascade.sens_reduce(kspace_pred_final,sens_maps)
        
        image_out = fastmri.complex_abs(image_prior_raw)
        return image_out, tissue_maps, last_p 


class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module,with_tissue=False):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()
        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1)*0.75)
        self.prior_weight = nn.Parameter(torch.ones(1)*1.0)
        self.with_tissue = with_tissue

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fftc.fft2c_new(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.complex_mul(
            fftc.ifft2c_new(x), fastmri.complex_conj(sens_maps)
        ).sum(dim=1, keepdim=True)

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor
    ) -> torch.Tensor:
        
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
        x_input = self.sens_reduce(current_kspace, sens_maps)
        
        if self.model.with_physics:
            tissue_maps, scan_params = self.model(x_input)
            t1 = (torch.sigmoid(tissue_maps[:, 0:1, :, :]) + 1e-4) * T1_MAX
            t2 = (torch.sigmoid(tissue_maps[:, 1:2, :, :]) + 1e-4) * T2_MAX
            pd = torch.sigmoid(tissue_maps[:, 2:3, :, :]) + 1e-4 
            tissue_maps_final = torch.cat([t1, t2, pd], dim=1)

            tr = torch.clamp(scan_params[:, 0:1], min=1e-3) * MAX_TR
            te = torch.clamp(scan_params[:, 1:2], min=1e-3) * MAX_TE
            alpha_deg = torch.clamp(scan_params[:, 2:3], min=1e-3) * MAX_ALPHA
            scan_params_final = torch.cat([tr, te, alpha_deg], dim=1)

            return tissue_maps_final, scan_params_final
        
        elif not self.model.with_physics:
            x_output, _ = self.model(x_input)
            model_term = self.sens_expand(x_output,sens_maps)
            return current_kspace - soft_dc - model_term

    @staticmethod
    def apply_bloch_equation(t1, t2, pd, tr, te, fa_rad):
        """
        Calculates the SPGR signal intensity.
        t1, t2, pd: maps [B, 1, H, W]
        tr, te, fa_deg: scalars or tensors [B, 1, 1, 1]
        """
        #safety
        t1 = torch.clamp(t1, min=10.0, max=5000.0)
        t2 = torch.clamp(t2, min=2.0, max=1000.0)
        pd = torch.clamp(pd, min=1e-4, max=10.0)

        e1 = torch.exp(-tr / (t1 + 1e-6))
        e2 = torch.exp(-te / (t2 + 1e-6))
        
        cos_fa = torch.cos(fa_rad)
        sin_fa = torch.sin(fa_rad)
        
        num = pd * sin_fa * (1 - e1)
        den = 1 - e1 * cos_fa
        den = torch.clamp(den, min=1e-4)
        
        signal = torch.abs((num / den) * e2)
        
        if torch.isnan(signal).any():
            signal = torch.where(torch.isnan(signal), torch.zeros_like(signal), signal)

        return (num / den) * e2


def debug_plot(tensor, title="Debug Plot"):
    with torch.no_grad():
        if torch.is_tensor(tensor):
            img = tensor.detach().cpu()
        else:
            img = tensor

        if img.shape[-1] == 2:
            img = torch.view_as_complex(img) if img.dtype != torch.complex64 else img
            img = torch.abs(img)
        elif img.dtype == torch.complex64 or img.dtype == torch.complex128:
            img = torch.abs(img)
            
        while img.ndim > 2:
            img = img[0]
            
        img_np = img.numpy() if hasattr(img, 'numpy') else img

        plt.figure(figsize=(5, 5))
        plt.imshow(img_np, cmap='gray')
        plt.title(f"{title}\nShape: {tensor.shape}")
        plt.colorbar()
        plt.show()
