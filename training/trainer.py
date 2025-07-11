from pathlib import Path
from torch.utils.data import DataLoader
import torch as torch
import deepinv as dinv
from models.operators import CmfOperator, ProbeParam
import numpy as np
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.unfolded import DEQ_builder
def build_trainer(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Chargement des 3 splits
    #ds_train = HDF5Dataset(config['data']['path'], split='train')
    #ds_valid = HDF5Dataset(config['data']['path'], split='valid')
    #ds_test  = HDF5Dataset(config['data']['path'], split='test')
    ds_train = dinv.datasets.HDF5Dataset(path=config['data']['path'], split='train')
    ds_valid = dinv.datasets.HDF5Dataset(path=config['data']['path'], split='valid')
    ds_test = dinv.datasets.HDF5Dataset(path=config['data']['path'], split='test')
    train_loader = DataLoader(ds_train,
                              batch_size=config['training']['training_batch_size'],
                              shuffle=True,
                              num_workers=config['data']['valid_num_workers'])
    valid_loader = DataLoader(ds_valid,
                              batch_size=config['training']['test_batch_size'],
                              shuffle=False,
                              num_workers=config['data']['num_workers'])
    test_loader  = DataLoader(ds_test,
                              batch_size=config['training']['batch_size'],
                              shuffle=False,
                              num_workers=config['data']['num_workers'])

    # 2) Récupère la fréquence à passer à l'opérateur
    frecon = config['model']['frequency']
    x=np.arange(-5,5.1,0.2)
    x=x*1e-3
    z=np.arange(60,70.1,0.2)
    z=z*1e-3
    xax, zax = np.meshgrid(x, z, indexing='xy')
    xax = torch.from_numpy(xax).to(device)
    zax = torch.from_numpy(zax).to(device)

    # 3) Construit opérateur et modèle
    physics = CmfOperator(frecon, ProbeParam(device), xax, zax, device=device)

    # Select the data fidelity term
    data_fidelity = L2()
    # Set up the trainable denoising prior. Here the prior model is common for all iterations. We use here a pretrained denoiser.
    prior = PnP(denoiser=dinv.models.DnCNN(depth=20, pretrained="download", in_channels=1, out_channels=1).to(device))

    # Unrolled optimization algorithm parameters
    max_iter = config['model']['max_iter'] if torch.cuda.is_available() else 10
    stepsize = config['training']['stepsize'] #
    sigma_denoiser = config['training']['sigma_denoiser']  # noise level parameter of the denoiser
    jacobian_free = False  # does not perform Jacobian inversion.

    params_algo = {  # wrap all the restoration parameters in a 'params_algo' dictionary
        "stepsize": stepsize,
        "g_param": sigma_denoiser,
    }
    trainable_params = [
        "stepsize",
        "g_param",
    ]  # define which parameters from 'params_algo' are trainable

    # Define the unfolded trainable model.
    model = DEQ_builder(
        iteration="PGD",  # For now DEQ is only possible with PGD, HQS and GD optimization algorithms.
        params_algo=params_algo.copy(),
        trainable_params=trainable_params,
        data_fidelity=data_fidelity,
        max_iter=max_iter,
        prior=prior,
        anderson_acceleration=True,
        anderson_acceleration_backward=True,
        history_size_backward=3,
        history_size=3,
        max_iter_backward=20,
        jacobian_free=jacobian_free,
    )

    # 4) Optimizer, scheduler, losses...

    epochs = config['training']['epochs'] if torch.cuda.is_available() else 2
    learning_rate = config['training']['learning_rate']

    # choose optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8))

    # choose supervised training loss
    losses = [dinv.loss.SupLoss(metric=dinv.metric.MSE())]
    save_path = config.get('save_path', 'results')
    Path(save_path).mkdir(parents=True, exist_ok=True)
    # 5) Instancie le Trainer avec train_loader et valid_loader
    trainer = dinv.Trainer(
        model=model,
        physics=physics,
        epochs=config['training']['epochs'],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        losses=losses,
        train_dataloader=train_loader,
        eval_dataloader=valid_loader,
        save_path=save_path,
        verbose=config['training'].get('verbose', True),
        wandb_vis=config.get('wandb', {}).get('enabled', False)
    )

    # 6) On stocke test_loader pour l'export plus tard
    trainer._test_loader = test_loader

    return trainer