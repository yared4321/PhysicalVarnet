import pathlib
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from fastmri_custom.pl_modules.varnet_module import VarNetModule
from fastmri_custom.data import mri_data, subsample, transforms
import fastmri_custom as fastmri
import yaml

# --- 1. Infrastructure and Data Functions ---

def load_config(config_path="config.yaml"):
    """Loads the configuration file."""
    if not os.path.exists(config_path):
        return {
            "paths": {
                "val_path": "./data/val/",
                "default_checkpoint": "checkpoints_final_changed/last_model_full.pth"
            }
        }
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_environment():
    """Configures system paths and returns the device."""
    current_dir = pathlib.Path.cwd()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    if os.path.abspath("./fastmri_custom") not in sys.path:
        sys.path.insert(0, os.path.abspath("./fastmri_custom"))
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_dataloader(data_path, batch_size=1):
    """Initializes the Dataset and DataLoader with the physical transform."""
    mask_func = subsample.RandomMaskFunc(center_fractions=[0.08], accelerations=[4])
    physics_transform = transforms.VarNetDataTransform(mask_func=mask_func)
    dataset = mri_data.SliceDataset(root=pathlib.Path(data_path), transform=physics_transform, challenge='multicoil')
    return torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)

def load_batches_to_device(iterator, num_batches, device):
    """Loads batches from the iterator directly to the GPU."""
    batch_list = []
    for _ in range(num_batches):
        try:
            batch = next(iterator)
            b_dev = SimpleNamespace(
                masked_kspace=batch[0].to(device),
                mask=batch[1].to(device),
                num_low_frequencies=int(batch[2][0].item()),
                target=batch[3].to(device),
                tr=batch[8].to(device), te=batch[9].to(device), alpha=batch[10].to(device),
                contrast=batch[11].to(device).to(torch.float32),
                t1_gt=batch[12].to(device), t2_gt=batch[13].to(device), pd_gt=batch[14].to(device)
            )
            batch_list.append(b_dev)
        except StopIteration:
            break
    return batch_list

def release_batches_from_gpu(batch_list):
    """Deletes batches and clears memory."""
    for b in batch_list:
        for attr in list(vars(b).keys()):
            delattr(b, attr)
    batch_list.clear()
    torch.cuda.empty_cache()

# --- 2. Physical Logic and Inference ---

def load_model(checkpoint_path, device):
    model = VarNetModule()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def run_physical_inference(model, batch, device):
    """Runs the model and produces reconstruction, maps, and simulation."""
    with torch.no_grad():
        recon, pred_maps, physics_p = model(batch.masked_kspace, batch.mask, batch.num_low_frequencies)
        
        tr_val = physics_p[:, 0:1].view(-1, 1, 1, 1)
        te_val = physics_p[:, 1:2].view(-1, 1, 1, 1)
        fa_rad = physics_p[:, 2:3].view(-1, 1, 1, 1)

        simulated = VarNetModule.apply_bloch_equation(
            pred_maps[:, 0:1], pred_maps[:, 1:2], pred_maps[:, 2:3],
            tr_val, te_val, fa_rad, batch.contrast
        )
    return recon, pred_maps, physics_p, simulated

# --- 3. Visualization ---

def visualize_all(batch, recon, maps, simulated, physics_p, index=0):
    """Displays all metrics: images, maps, and differences."""
    target_np = batch.target[index].cpu().numpy()
    recon_np = recon[index].squeeze().cpu().numpy()
    sim_np = simulated[index].squeeze().cpu().numpy()
    
    _, recon_np = transforms.center_crop_to_smallest(target_np, recon_np)
    _, sim_np = transforms.center_crop_to_smallest(target_np, sim_np)
    diff_abs = np.abs(target_np - sim_np)

    # Figure 1: Reconstruction and Simulation Comparison
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    vmax = np.percentile(target_np, 99)
    
    axes[0].imshow(recon_np, cmap='gray', vmax=vmax); axes[0].set_title("Model Recon")
    axes[1].imshow(sim_np, cmap='gray', vmax=vmax); axes[1].set_title("Bloch Sim")
    axes[2].imshow(target_np, cmap='gray', vmax=vmax); axes[2].set_title("Ground Truth")
    im = axes[3].imshow(diff_abs, cmap='inferno', vmax=vmax*0.2)
    axes[3].set_title(f"Abs Diff (Mean: {np.mean(diff_abs):.4f})")
    plt.colorbar(im, ax=axes[3]); plt.tight_layout(); plt.show()

    # Figure 2: Display cropped tissue maps (T1, T2, PD)
    map_data = maps[index].cpu().numpy() # [3, H, W]
    
    cropped_maps = []
    for i in range(map_data.shape[0]):
        _, cropped = transforms.center_crop_to_smallest(target_np, map_data[i])
        cropped_maps.append(cropped)
    
    fig2, ax2 = plt.subplots(1, 3, figsize=(18, 5))
    cf = [('T1 Map (ms)', 3000, 'magma'), ('T2 Map (ms)', 500, 'viridis'), ('PD Map', 1, 'gray')]
    
    for i, (title, mx, cm) in enumerate(cf):
        im = ax2[i].imshow(cropped_maps[i], cmap=cm, vmin=0, vmax=mx)
        ax2[i].set_title(title)
        plt.colorbar(im, ax=ax2[i])
    
    plt.tight_layout()
    plt.show()

    # Physics Check
    print(f"\n--- Physics Check ---")
    print(f"Pred -> TR: {physics_p[index,0]:.1f} | TE: {physics_p[index,1]:.1f} | Alpha: {physics_p[index,2]:.1f}")
    print(f"True -> TR: {batch.tr[index]:.1f} | TE: {batch.te[index]:.1f} | Alpha: {batch.alpha[index]:.1f}")
    
    print(f"Mean T1: {cropped_maps[0].mean():.1f}ms | Mean T2: {cropped_maps[1].mean():.1f}ms | Mean PD: {cropped_maps[2].mean():.3f}")

# --- 4. Main ---

def main():
    device = setup_environment()
    config = load_config()
    
    val_path = config['paths'].get('val_path', './data/val/')
    checkpoint_path = config['paths'].get('default_checkpoint', 'checkpoints_final_changed/last_model_full.pth')

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint missing: {checkpoint_path}"); return

    model = load_model(checkpoint_path, device)
    val_loader = get_dataloader(val_path)
    val_iter = iter(val_loader)
    
    # Load batch set
    batch_list = load_batches_to_device(val_iter, 4, device)
    
    if batch_list:
        for idx in range(len(batch_list)):
            recon, maps, params, sim = run_physical_inference(model, batch_list[idx], device)
            visualize_all(batch_list[idx], recon, maps, sim, params)
            
        del recon, maps, params, sim
        release_batches_from_gpu(batch_list)

if __name__ == "__main__":
    main()