import argparse, yaml, os
from scipy.io import savemat
import wandb
from training.trainer import build_trainer

def export_test(trainer):
    out = os.path.join(trainer.save_path, 'test_results')
    os.makedirs(out, exist_ok=True)
    for i, batch in enumerate(trainer._test_loader):
        x, csm, gt = batch['x'], batch['csm'], batch['gt']
        pred = trainer.model(x.to(trainer.device)).detach().cpu().numpy()
        savemat(os.path.join(out, f'{i:03d}.mat'),
                {'x':x.numpy(),'csm':csm.numpy(),'gt':gt.numpy(),'pred':pred})
        if trainer.wandb_vis and i<5:
            wandb.log({
                'test/pred': wandb.Image(pred[0], caption='pred'),
                'test/gt':   wandb.Image(gt[0],   caption='gt')
            })

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-c','--config',required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if cfg.get('wandb',{}).get('enabled',False):
        wandb.login(); wandb.init(project=cfg['project'],config=cfg)
    trainer = build_trainer(cfg)
    trainer.train()
    export_test(trainer)

if __name__=='__main__': main()
