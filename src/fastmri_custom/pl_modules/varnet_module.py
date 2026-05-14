"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import matplotlib.pyplot as plt
import torch
import numpy as np

from argparse import ArgumentParser
import math
import torch

import fastmri_custom as fastmri # Use local directory as an alias
import fastmri_custom.fftc as fftc # Important for fft2c/ifft2c
from ..data import transforms
from ..models.varnet import VarNet, VarNetBlock

from .mri_module import MriModule
import torch.nn.functional as F

class VarNetModule(MriModule):
    """
    VarNet training module.

    This can be used to train variational networks from the paper:

    A. Sriram et al. End-to-end variational networks for accelerated MRI
    reconstruction. In International Conference on Medical Image Computing and
    Computer-Assisted Intervention, 2020.

    which was inspired by the earlier paper:

    K. Hammernik et al. Learning a variational network for reconstruction of
    accelerated MRI data. Magnetic Resonance inMedicine, 79(6):3055–3071, 2018.
    """
    # ערכים גלובליים של הסקאלה הפיזיקלית (לפי ה-LUT והמטא-דאטה)
    MAX_TR = 10000.0
    MAX_TE = 200.0
    MAX_ALPHA = 180.0
    params_scale = torch.tensor([MAX_TR, MAX_TE, 2*math.pi])
    
    # Brainweb values for map scaling (T1, T2, PD)
    T1_MAX = 3000.0 
    T2_MAX = 500.0
    PD_MAX = 1.0
    tissue_scale = torch.tensor([T1_MAX, T2_MAX, PD_MAX])

    def __init__(
        self,
        num_cascades: int = 12,
        pools: int = 4,
        chans: int = 18,
        sens_pools: int = 4,
        sens_chans: int = 8,
        lr: float = 1e-3,
        lr_step_size: int = 5,
        lr_gamma: float = 0.5,
        weight_decay: float = 1e-5,
        **kwargs,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            chans: Number of channels for cascade U-Net.
            sens_pools: Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            sens_chans: Number of channels for sensitivity map U-Net.
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
            num_sense_lines: Number of low-frequency lines to use for sensitivity map
                computation, must be even or `None`. Default `None` will automatically
                compute the number from masks. Default behaviour may cause some slices to
                use more low-frequency lines than others, when used in conjunction with
                e.g. the EquispacedMaskFunc defaults. To prevent this, either set
                `num_sense_lines`, or set `skip_low_freqs` and `skip_around_low_freqs`
                to `True` in the EquispacedMaskFunc. Note that setting this value may
                lead to undesired behaviour when training on multiple accelerations
                simultaneously.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.pools = pools
        self.chans = chans
        self.sens_pools = sens_pools
        self.sens_chans = sens_chans
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.varnet = VarNet(
            num_cascades=self.num_cascades,
            sens_chans=self.sens_chans,
            sens_pools=self.sens_pools,
            chans=self.chans,
            pools=self.pools,
        )

        self.loss = fastmri.SSIMLoss()

    def forward(self, masked_kspace, mask, num_low_frequencies):
        return self.varnet(masked_kspace, mask, num_low_frequencies)

    def training_step(self, batch, batch_idx):
        has_physics = batch.t1_gt.ndim > 1
        
        # 1. Standard Forward Pass
        image_prior, pred_maps, pred_params = self.varnet(batch.masked_kspace,
                                                        batch.mask,
                                                        batch.num_low_frequencies
                                                        )      
        
        image_bloch_tissue = self.apply_bloch_equation(
                                                t1=pred_maps[:, 0:1].float(), 
                                                t2=pred_maps[:, 1:2].float(), 
                                                pd=pred_maps[:, 2:3].float(),
                                                tr=batch.tr.float().detach().view(-1, 1, 1, 1), 
                                                te=batch.te.float().detach().view(-1, 1, 1, 1), 
                                                fa_deg=batch.alpha.float().detach().view(-1, 1, 1, 1),
                                                gt_contrast=batch.contrast.float().view(-1, 1, 1, 1)).float()

        image_bloch_params = self.apply_bloch_equation(
                                                t1=batch.t1_gt.float().detach(), 
                                                t2=batch.t2_gt.float().detach(), 
                                                pd=batch.pd_gt.float().detach(),
                                                tr=pred_params[:, 0:1].float().view(-1, 1, 1, 1), 
                                                te=pred_params[:, 1:2].float().view(-1, 1, 1, 1), 
                                                fa_deg=pred_params[:, 2:3].float().view(-1, 1, 1, 1),
                                                gt_contrast=batch.contrast.view(-1, 1, 1, 1)).float()
        
        _, image_bloch_tissue = transforms.center_crop_to_smallest(batch.target, image_bloch_tissue.squeeze(1))
        _, image_bloch_params = transforms.center_crop_to_smallest(batch.target, image_bloch_params.squeeze(1))
        
        dr = torch.tensor([1.0], dtype=torch.float32, device=batch.target.device)
        
        if has_physics:
            image_bloch_gt = self.apply_bloch_equation(
                                                t1= batch.t1_gt.float(), 
                                                t2= batch.t2_gt.float(), 
                                                pd= batch.pd_gt.float(),
                                                tr= batch.tr.float().view(-1, 1, 1, 1), 
                                                te= batch.te.float().view(-1, 1, 1, 1), 
                                                fa_deg= batch.alpha.float().view(-1, 1, 1, 1),
                                                gt_contrast=batch.contrast.float().view(-1, 1, 1, 1)).float()
                                                
            _, image_bloch_gt = transforms.center_crop_to_smallest(batch.target, image_bloch_gt.squeeze(1))

            ssim_bloch_tissue_param = 0.3
            l1_bloch_tissue_param = 0.7

            tissue_scale = torch.tensor([self.T1_MAX, self.T2_MAX, self.PD_MAX], 
                                        device=pred_maps.device).view(1, 3, 1, 1)
            pred_maps_norm = pred_maps / tissue_scale
            t1_gt_norm = batch.t1_gt / self.T1_MAX
            t2_gt_norm = batch.t2_gt / self.T2_MAX
            pd_gt_norm = batch.pd_gt / self.PD_MAX
            
            pred_maps_norm, _ = transforms.center_crop_to_smallest(pred_maps_norm, t1_gt_norm)
            
            ssim_bloch_tissue = self.loss(image_bloch_tissue, image_bloch_gt, data_range=dr)
            l1_bloch_tissue = F.l1_loss(image_bloch_tissue, image_bloch_gt)
            l1_bloch_params = F.l1_loss(image_bloch_params, image_bloch_gt)
            
            ssim_t1 = self.loss(pred_maps_norm[:, 0:1,:,:], t1_gt_norm, data_range=dr)
            l1_t1 = F.l1_loss(pred_maps_norm[:, 0:1,:,:], t1_gt_norm)

            ssim_t2 = self.loss(pred_maps_norm[:, 1:2,:,:], t2_gt_norm, data_range=dr)
            l1_t2 = F.l1_loss(pred_maps_norm[:, 1:2,:,:], t2_gt_norm)

            ssim_pd = self.loss(pred_maps_norm[:, 2:3,:,:], pd_gt_norm, data_range=dr)
            l1_pd = F.l1_loss(pred_maps_norm[:, 2:3,:,:], pd_gt_norm)

            loss_t1 = 0.5*l1_t1 + 0.5*ssim_t1
            loss_t2 = 0.5*l1_t2 + 0.5*ssim_t2
            loss_pd = 0.5*l1_pd + 0.5*ssim_pd
            loss_maps = loss_t1 + loss_t2 + loss_pd
        else:
            loss_maps = 0.0
            ssim_bloch_tissue_param = 0.2
            l1_bloch_tissue_param = 0.2
            l1_bloch_params = 0.0
            l1_bloch_tissue = 0.0
            ssim_bloch_tissue = 0.0

        _, image_prior_reshaped = transforms.center_crop_to_smallest(batch.target, image_prior.squeeze(1))

        vmax = (batch.target.max() + 1e-9).float()
        target_norm = batch.target / vmax
        image_norm = image_prior_reshaped / vmax

        input_image = image_norm.unsqueeze(1)
        input_target = target_norm.unsqueeze(1)     
        
        bloch_loss = (ssim_bloch_tissue_param * ssim_bloch_tissue) + (l1_bloch_tissue_param * l1_bloch_tissue) 

        ssim_image = self.loss(input_image, input_target, data_range=dr)
        l1_image = F.l1_loss(input_image, input_target)
        image_loss = 0.5*ssim_image + 0.5*l1_image
        
        params_scale = torch.tensor([self.MAX_TR, self.MAX_TE, self.MAX_ALPHA], 
                                    device=pred_params.device).view(1, 3)
        true_params = torch.stack([batch.tr, batch.te, batch.alpha], dim=1).float()
        
        pred_norm = pred_params / params_scale
        true_norm = true_params / params_scale
        
        loss_tr_te = F.l1_loss(pred_norm[:, :2], true_norm[:, :2])
        loss_alpha = F.l1_loss(pred_norm[:, 2], true_norm[:, 2])

        loss_params = (0.5 * loss_tr_te) + (0.5 * loss_alpha) + (0.5*l1_bloch_params)

        total_loss = 2.0 * bloch_loss + 10.0 * image_loss + 5.0 * loss_maps + 10.0 * loss_params 

        physics_weight_norm = torch.norm(torch.stack([torch.norm(p) for p in self.varnet.cascades[-1].model.unet.physics_head.parameters()]))
        tissue_weight_norm = torch.norm(torch.stack([torch.norm(p) for p in self.varnet.cascades[-1].model.unet.parameters()]))
        image_weight_norm = torch.norm(torch.stack([torch.norm(p) for p in self.varnet.cascades[-2].model.unet.parameters()]))

        with torch.no_grad():
            print(f"--- Step Info ---")
            print(f"physics head Weight Norm: {physics_weight_norm:.4f}")      
            print(f"tissue head Weight Norm: {tissue_weight_norm:.4f}")      
            print(f"image head Weight Norm: {image_weight_norm:.4f}")      

            print(f"Bloch Loss: {bloch_loss:.4f} : ssim_bloch_tissue:{ssim_bloch_tissue:.4f}| l1_bloch_tissue: {l1_bloch_tissue:.4f}| l1_bloch_params:{l1_bloch_params:.4f}")
            print(f"Image Loss: {image_loss:.4f} : ssim_image: {ssim_image:.4f}| l1_image: {l1_image:.4f}")
            print(f"PHYS LOSS: {loss_params.item():.4f} | TR: {pred_params[0,0].item():.0f}/{batch.tr[0].item():.0f} | TE: {pred_params[0,1].item():.1f}/{batch.te[0].item():.1f} | FA: {pred_params[0,2].item():.1f}/{batch.alpha[0].item():.1f}")
            
            if has_physics:
                print(f"loss_maps: {loss_maps:.4f}: loss_t1: {loss_t1:.4f}| loss_t2: {loss_t2:.4f}| loss_pd: {loss_pd:.4f}")
            else:
                print(f"loss_maps: 0.0000 (No Physics Data)")
            print(f"total loss: {total_loss:.4f}")

        self.log("train/image_loss", image_loss)
        self.log("train/loss_maps", loss_maps)
        self.log("train/loss_params", loss_params)
        self.log("train/total_loss", total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        output, physics_pred = self.forward(
            batch.masked_kspace, batch.mask, batch.num_low_frequencies
        )
        target, output = transforms.center_crop_to_smallest(batch.target, output)

        val_loss = self.loss(
            output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
        )

        self.log("val_loss", val_loss)

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output,
            "target": target,
            "val_loss": val_loss
        }

    def test_step(self, batch, batch_idx):
        output = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies)

        # check for FLAIR 203
        if output.shape[-1] < batch.crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])
        else:
            crop_size = batch.crop_size

        output = transforms.center_crop(output, crop_size)

        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "output": output.cpu().numpy(),
        }

    def configure_optimizers(self):
        backbone_params = [p for n, p in self.varnet.named_parameters() if 'physics_head' not in n and 'tissue_head' not in n]
        physics_params = [p for n, p in self.varnet.named_parameters() if 'physics_head' in n]
        tissue_params = [p for n, p in self.varnet.named_parameters() if 'tissue_head' in n]
        image_params = [p for n, p in self.varnet.named_parameters() if 'image_head' in n]

        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': 1e-3, 'weight_decay': 0},
            {'params': physics_params, 'lr': 1e-3, 'weight_decay': 0}, 
            {'params': tissue_params, 'lr': 1e-3, 'weight_decay': 0},
            {'params': image_params, 'lr': 1e-3, 'weight_decay': 0 }
        ])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/total_loss",
                "interval": "epoch",
                "frequency": 1
            },
        }

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # param overwrites

        # network params
        parser.add_argument(
            "--num_cascades",
            default=12,
            type=int,
            help="Number of VarNet cascades",
        )
        parser.add_argument(
            "--pools",
            default=4,
            type=int,
            help="Number of U-Net pooling layers in VarNet blocks",
        )
        parser.add_argument(
            "--chans",
            default=18,
            type=int,
            help="Number of channels for U-Net in VarNet blocks",
        )
        parser.add_argument(
            "--sens_pools",
            default=4,
            type=int,
            help="Number of pooling layers for sense map estimation U-Net in VarNet",
        )
        parser.add_argument(
            "--sens_chans",
            default=8,
            type=float,
            help="Number of channels for sense map estimation U-Net in VarNet",
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.0003, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.1,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser
    
    @staticmethod
    def total_variation_loss(img):
        """
        Encourages edge formation and prevents uniform color blocks in maps.
        """
        pixel_dif_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
        pixel_dif_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        return pixel_dif_h.sum() + pixel_dif_w.sum()

    @staticmethod
    def apply_bloch_equation(t1, t2, pd, tr, te, fa_deg, gt_contrast):
        """
        Calculates the SPGR signal intensity.
        t1, t2, pd: maps [B, 1, H, W]
        tr, te, fa_deg: scalars or tensors [B, 1, 1, 1]
        """        
        alpha_rad = fa_deg * (torch.pi / 180.0)
        e1 = torch.exp(-tr / (t1 + 1e-6))
        e2 = torch.exp(-te / (t2 + 1e-6))
        
        cos_fa = torch.cos(alpha_rad)
        sin_fa = torch.sin(alpha_rad)
        
        num = pd * sin_fa * (1 - e1)
        den = 1 - e1 * cos_fa
        den = torch.clamp(den, min=1e-4)
       
        num = (1 - e1) * torch.sin(alpha_rad)
        den = 1 - (torch.cos(alpha_rad) * e1)
        res_t1 = pd * (num / (den + 1e-10)) * e2
    
        alpha_effect = torch.pow(torch.sin(alpha_rad / 2), 3)
        res_t2 = pd * (1 - e1) * e2 * alpha_effect
        
        signal = torch.where(gt_contrast < 0.4, res_t1, res_t2)
        return signal

def debug_plot(tensor, title="Debug Plot"):
    """
    Displays tensor at any stage (complex, batch, image, or k-space).
    """
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
