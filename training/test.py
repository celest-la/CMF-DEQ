# training/test.py
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from scipy.io import savemat
import os

import deepinv as dinv
from deepinv.datasets import HDF5Dataset
from models.operators import CmfOperator, ProbeParam

def build_test_components(config):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoader test
    ds_test = HDF5Dataset(config['data']['path'], split='test')
    test_loader = DataLoader(
        ds_test,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )

    # Physique (avec freq tirée du dataset)
    freq = ds_test.frequency
    probe = ProbeParam(device=device)
    physics = CmfOperator(
        frecon=config['model']['frecon'],
        h=probe,
        X=config['model']['X'],
        Z=config['model']['Z'],
        frequency=freq,
        device=device
    )

    # Modèle DEQ
    prior = dinv.optim.prior.PnP(
        denoiser=dinv.models.DnCNN(
            depth=config['model']['dncnn_depth'],
            pretrained="download",
            in_channels=1,
            out_channels=1
        ).to(device)
    )
    model = dinv.unfolded.DEQ_builder(
        iteration="PGD",
        params_algo=config['model']['params_algo'],
        trainable_params=config['model']['trainable_params'],
        data_fidelity=dinv.optim.data_fidelity.L2(),
        max_iter=config['model']['max_iter'],
        prior=prior,
        anderson_acceleration=True,
        anderson_acceleration_backward=True,
        history_size=config['model']['history_size'],
        history_size_backward=config['model']['history_size_backward'],
        max_iter_backward=config['model']['max_iter_backward'],
        jacobian_free=config['model']['jacobian_free']
    ).to(device)

    return device, test_loader, physics, model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config', required=True,
                        help='Chemin vers configs/train.yaml')
    parser.add_argument('-k','--ckpt', required=True,
                        help='Chemin vers le checkpoint .pth final')
    args = parser.parse_args()

    # 1) Charge config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 2) Reconstruit test_loader, physics, modèle
    device, test_loader, physics, model = build_test_components(config)

    # 3) Charge les poids
    checkpoint = torch.load(args.ckpt, map_location=device)
    # si tu as sauvé state_dict sous 'model_state_dict':
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 4) (Optionnel) init wandb pour le test
    wandb_vis = config.get('wandb', {}).get('enabled', False)
    if wandb_vis:
        import wandb
        wandb.login()
        wandb.init(project=config['project'],
                   name=f"{config.get('run_name','test')}_inference",
                   config=config)

    # 5) Dossier de sortie
    out_dir = os.path.join(config.get('save_path','results'), 'test_results')
    os.makedirs(out_dir, exist_ok=True)

    # 6) Boucle d'inférence
    for i, batch in enumerate(test_loader):
        x   = batch['x'].to(device)
        csm = batch['csm'].to(device)
        gt  = batch['gt'].to(device)

        with torch.no_grad():
            pred = model(x)

        # 6a) Sauvegarde .mat
        savemat(
            os.path.join(out_dir, f'sample_{i:03d}.mat'),
            {
                'x':   x.cpu().numpy(),
                'csm': csm.cpu().numpy(),
                'gt':  gt.cpu().numpy(),
                'pred': pred.detach().cpu().numpy()
            }
        )


if __name__ == '__main__':
    main()
