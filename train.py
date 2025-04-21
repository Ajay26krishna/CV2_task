from data.dataloader import get_dataloader
import torch
import numpy as np
import os
from omegaconf import OmegaConf
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.body_model.body_model import BodyModel
from os import path as osp
from models.transformer import Transformer, Config


#Loading SMPLx Body Model

support_dir = r'C:\Users\shasi\Downloads\cv2_term_project\VPoserModelFiles\\'
bm_fname =  osp.join(support_dir,'smplx_neutral_model.npz')   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device is', device)

bm = BodyModel(bm_fname=bm_fname).to(device)
expr_dir = osp.join(support_dir,'vposer_v2_05/') #'TRAINED_MODEL_DIRECTORY'


# ==== Load VPoser from .ckpt with manually created config ====


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




train_data_dir = r'C:\Users\shasi\Downloads\cv2_term_project\data\AMASS_CMUsubset\test'
test_data_dir = r'C:\Users\shasi\Downloads\cv2_term_project\data\AMASS_CMUsubset\train'
# dataloader = get_dataloader(
#     data_dir=data_root,
#     vp_model=vp,
#     device=device,
#     seq_len=64,
#     pred_step=1,
#     batch_size=32
# )


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = Config()
model = Transformer(config).to(device)

train_loader = get_dataloader(train_data_dir, vp, device, seq_len=64, pred_step=1, batch_size=32)
val_loader = get_dataloader(test_data_dir, vp, device, seq_len=64, pred_step=1, batch_size=32, shuffle=False)

for batch in train_loader:
    x, y = batch
    x, y = x.to(device), y.to(device)
    pred, loss = model(x, y)
    print("Train batch loss:", loss.item())
    break



