# dataloader.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class PoseSequenceDataset(Dataset):
    def __init__(self, latent_seq, seq_len, pred_step):
        self.inputs = []
        self.targets = []
        for i in range(len(latent_seq) - seq_len - pred_step):
            self.inputs.append(latent_seq[i:i + seq_len])
            self.targets.append(latent_seq[i + seq_len:i + seq_len + pred_step])
        self.inputs = torch.stack(self.inputs)
        self.targets = torch.stack(self.targets).squeeze(1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def load_all_latents(data_dir, vp_model, device):
    all_latents = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith('.npz'):
            poses = np.load(os.path.join(data_dir, fname))['poses'][:, 3:66]  # remove global rotation
            poses = torch.from_numpy(poses).float().to(device)
            latents = vp_model.encode(poses).mean  # (T, 32)
            all_latents.append(latents.detach().cpu())
    latent_data = torch.cat(all_latents, dim=0)
    print('Combined latent shape:', latent_data.shape)
    return latent_data

def get_dataloader(data_dir, vp_model, device, seq_len, pred_step, batch_size, shuffle=True):
    latent_seq = load_all_latents(data_dir, vp_model, device)
    dataset = PoseSequenceDataset(latent_seq, seq_len=seq_len, pred_step=pred_step)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
