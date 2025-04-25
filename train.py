from data.dataloader import get_dataloader
import torch
import numpy as np
import os
from omegaconf import OmegaConf
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.body_model.body_model import BodyModel
from os import path as osp
from models.transformer import Transformer, Config
from torch.optim import AdamW
from tqdm import tqdm
from evaluate import run_evaluation


def train_epoch(model, dataloader, optimizer, device, vposer, smpl):
    model.train()
    total_loss = 0.0
    loop = tqdm(dataloader, desc="Training", leave=False)

    for x, latent_targets, pose_targets in loop:
        x, latent_targets, pose_targets = x.to(device), latent_targets.to(device), pose_targets.to(device)

        optimizer.zero_grad()
        pred_latents, _ = model(x)  # ignore latent loss

        # Decode predicted latents to body pose using VPoser
        pred_pose = vposer.decode(pred_latents[:, -1])['pose_body']  # (B, 63)

        # Flatten and get joints from SMPL
        smpl_output = smpl(body_pose=pred_pose)
        pred_joints = smpl_output.joints[:, :24]  # (B, 24, 3)

        # Get ground truth joints from pose_targets via SMPL
        with torch.no_grad():
            gt_smpl = smpl(body_pose=pose_targets[:, 0])  # assume 1-step target
            gt_joints = gt_smpl.joints[:, :24]

        # Compute MPJPE loss
        loss = torch.norm(pred_joints - gt_joints, dim=-1).mean()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


def eval_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    loop = tqdm(dataloader, desc="Validation", leave=False)
    with torch.no_grad():
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)
#Loading SMPLx Body Model
#filenames for loading VPoser VAE network, neutral SMPL body model, AMASS sample data

from os import path as osp

support_dir = '/content/gdrive/MyDrive/VPoserModelFiles/'

expr_dir = osp.join(support_dir,'vposer_v2_05/') #'TRAINED_MODEL_DIRECTORY'
bm_fname =  osp.join(support_dir,'smplx_neutral_model.npz')    #'PATH_TO_SMPLX_model.npz'  neutral smpl body model
sample_amass_fname = osp.join(support_dir, 'amass_sample.npz')  # a sample npz file from AMASS


print(expr_dir)
print(bm_fname)
print(sample_amass_fname)

# ==== Load VPoser from .ckpt with manually created config ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device is', device)

from human_body_prior.body_model.body_model import BodyModel
bm = BodyModel(bm_fname=bm_fname).to(device)

#Loading VPoser VAE Body Pose Prior
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

vp, ps = load_model(expr_dir, model_code=VPoser,
                              remove_words_in_model_weights='vp_model.',
                              disable_grad=True,
                              comp_device=device)
vp = vp.to(device)




train_data_dir =  '/content/gdrive/MyDrive/AMASS_CMUsubset/test/'
test_data_dir = '/content/gdrive/MyDrive/AMASS_CMUsubset/train/'

def main():
    # Configuration
    seq_len = 64
    pred_step = 1
    batch_size = 64
    epochs = 10
    lr = 1e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and data
    config = Config()
    model = Transformer(config).to(device)

    train_loader = get_dataloader(train_data_dir, vp, device, seq_len, pred_step, batch_size, shuffle=True)
    val_loader = get_dataloader(test_data_dir, vp, device, seq_len, pred_step, batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, vp, bm)
        val_loss = eval_epoch(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "latent_transformer.pth")
    print("Training complete and model saved.")
    print('running evaluation')
    run_evaluation(model,test_data_dir, vp, bm, device, SEQ_LEN=64)


if __name__ == '__main__':
    main()