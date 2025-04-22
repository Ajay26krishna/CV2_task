import torch
import trimesh
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from data.dataloader import get_dataloader
import os
from datetime import datetime

def run_evaluation(model, test_data_dir, vp, bm, device, seq_len=64):

    os.makedirs("results", exist_ok=True)

    latent_data = get_dataloader(test_data_dir, vp, device, seq_len, 1, 1, shuffle=False)

    model.eval()
    with torch.no_grad():
        seed_seq = latent_data[100:100+seq_len].unsqueeze(0).to(device)
        prediction = model(seed_seq).squeeze(0)

    reco_pose = vp.decode(prediction.unsqueeze(0))['pose_body'].contiguous().view(-1, 63)
    reco_mesh = bm(pose_body=reco_pose)
    verts = c2c(reco_mesh.v)[0]
    mesh = trimesh.base.Trimesh(verts, c2c(bm.f))
    mesh.show()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mesh_path = os.path.join("results", f"predicted_mesh_{timestamp}.ply")
    mesh.export(mesh_path)
    print(f"Saved mesh to {mesh_path}")
    # Visualize
    # mesh.show()

# Example usage (make sure to call this with proper model and data)
# run_evaluation(model, latent_data, vp, bm, device, SEQ_LEN=64)
