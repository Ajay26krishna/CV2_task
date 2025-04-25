# dataloader.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class PoseSequenceDataset(Dataset):
    def __init__(self, latent_seq, pose_seq, seq_len, pred_step):
        self.inputs = []
        self.latent_targets = []
        self.pose_targets = []
        for i in range(len(latent_seq) - seq_len - pred_step):
            self.inputs.append(latent_seq[i:i + seq_len])
            self.latent_targets.append(latent_seq[i + seq_len:i + seq_len + pred_step])
            self.pose_targets.append(pose_seq[i + seq_len:i + seq_len + pred_step])
        self.inputs = torch.stack(self.inputs)
        self.latent_targets = torch.stack(self.latent_targets).squeeze(1)
        self.pose_targets = torch.stack(self.pose_targets).squeeze(1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.latent_targets[idx], self.pose_targets[idx]

def load_all_latents(data_dir, vp_model, device):
    all_latents = []
    all_poses = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith('.npz'):
            poses = np.load(os.path.join(data_dir, fname))['poses'][:, 3:66]  # remove global rotation
            poses = torch.from_numpy(poses).float().to(device)
            latents = vp_model.encode(poses).mean  # (T, 32)
            all_latents.append(latents.detach().cpu())
            all_poses.append(poses.detach().cpu())  # save poses too
    latent_data = torch.cat(all_latents, dim=0)
    pose_data = torch.cat(all_poses, dim=0)
    print('Combined latent shape:', latent_data.shape)
    return latent_data, pose_data


def get_dataloader(data_dir, vp_model, device, seq_len, pred_step, batch_size, shuffle=True):
    latent_seq, pose_seq = load_all_latents(data_dir, vp_model, device)
    dataset = PoseSequenceDataset(latent_seq, pose_seq, seq_len=seq_len, pred_step=pred_step)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
