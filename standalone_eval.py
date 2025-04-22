import torch
import torch.nn as nn
import trimesh
import os
import numpy as np
from datetime import datetime
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.body_model.body_model import BodyModel
from omegaconf import OmegaConf
from models.transformer import Transformer,Config
import os.path as osp


# ==== Load VPoser and SMPL ==== 
support_dir = r'C:\Users\shasi\Downloads\cv2_term_project\VPoserModelFiles\\'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bm_fname =  osp.join(support_dir,'smplx_neutral_model.npz')   


bm = BodyModel(bm_fname=bm_fname).to(device)

expr_dir = osp.join(support_dir,'vposer_v2_05/') #'TRAINED_MODEL_DIRECTORY'

ckpt_path = os.path.join(expr_dir, 'snapshots', 'V02_05_epoch=13_val_loss=0.03.ckpt')
ckpt = torch.load(ckpt_path, map_location=device)
vp_cfg = OmegaConf.create({
    'model_params': {
        'num_neurons': 512,
        'num_layers': 2,
        'latentD': 32
    }
})

vp = VPoser(vp_cfg)
vp.load_state_dict(ckpt['state_dict'], strict=False)
print(vp.eval())
vp.to(device)


# ==== Load Latent Data ====
data_dir = r'C:\Users\shasi\Downloads\cv2_term_project\data\AMASS_CMUsubset\test'
all_latents = []
for fname in sorted(os.listdir(data_dir)):
    if fname.endswith('.npz'):
        poses = np.load(os.path.join(data_dir, fname))['poses'][:, 3:66]  # remove global rotation
        poses = torch.from_numpy(poses).float().to(device)
        latents = vp.encode(poses).mean  # (T, 32)
        all_latents.append(latents.detach().cpu())
latent_data = torch.cat(all_latents, dim=0)

config = Config()
model = Transformer(config).to(device)
model.load_state_dict(torch.load(r'C:\Users\shasi\Downloads\cv2_term_project\latent_transformer.pth', map_location=device))

# ==== Evaluation ====
def run_evaluation(model, latent_data, vp, bm, device, SEQ_LEN=64, result_dir="results"):
    os.makedirs(result_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        seed_seq = latent_data[100:100+SEQ_LEN].unsqueeze(0).to(device)
        prediction = model(seed_seq)[0].squeeze(0)

    reco_pose = vp.decode(prediction.unsqueeze(0))['pose_body'].contiguous().view(-1, 63)
    reco_mesh = bm(pose_body=reco_pose)
    verts = reco_mesh.v[0].detach().cpu().numpy()
    faces = bm.f.cpu().numpy()
    mesh = trimesh.base.Trimesh(verts, faces)

    
    # n_pred_frames = prediction.shape[0]
    # original_pose = latent_data[100+SEQ_LEN:100+SEQ_LEN+n_pred_frames].to(device)
    # # breakpoint()1
    # original_mesh = bm(pose_body=original_pose)
    # verts_orig = original_mesh.v[0].detach().cpu().numpy()
    # mesh_orig = trimesh.base.Trimesh(verts_orig, faces, vertex_colors=[128, 0, 128, 255])  

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mesh_reco_path = os.path.join(result_dir, f"transformer_predicted_mesh_{timestamp}.ply")
    # mesh_orig_path = os.path.join(result_dir, f"original_mesh_{timestamp}.ply")
    mesh.export(mesh_reco_path)
    # mesh_orig.export(mesh_orig_path)
    print(f"Saved mesh to {mesh}")

run_evaluation(model, latent_data, vp, bm, device, SEQ_LEN=64)