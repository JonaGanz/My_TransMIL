import argparse
from pathlib import Path
import numpy as np
import glob

from datasets import DataInterface
from models import ModelInterface
from utils.utils import *

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer

import os

def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path',help='path containing model checkpoints')
    parser.add_argument('--config_path',help='path to config file')
        
    parser.add_argument('--k',help='number of folds',type=int, default=10)
    parser.add_argument('--k_start',help='start fold',type=int, default=None)
    parser.add_argument('--k_end', help='end fold',type=int, default=None)
    parser.add_argument('--gpus', default = [1])
    return parser.parse_args()


def main(checkpoint_path, cfg):
    
    #---->Initialize seed
    pl.seed_everything(cfg.General.seed)
    
    #---->Define Data 
    DataInterface_dict = {'train_batch_size': cfg.Data.train_dataloader.batch_size,
                'train_num_workers': cfg.Data.train_dataloader.num_workers,
                'test_batch_size': cfg.Data.test_dataloader.batch_size,
                'test_num_workers': cfg.Data.test_dataloader.num_workers,
                'dataset_name': cfg.Data.dataset_name,
                'dataset_cfg': cfg.Data,}
    dm = DataInterface(**DataInterface_dict)
    
    #---->Instantiate Trainer
    trainer = Trainer(
        num_sanity_val_steps=0,
        gpus=1,
        amp_level=cfg.General.amp_level,  
        precision=cfg.General.precision,  
        accumulate_grad_batches=cfg.General.grad_acc,
        deterministic=True,
        check_val_every_n_epoch=1,
    )
    # look for trained models under checkpoint_path/fold<k>/epoch<>.ckpt
    fold_path = Path(checkpoint_path) / f"fold{cfg.Data.fold}"
    model_name = [f for f in os.listdir(fold_path) if f.endswith('.ckpt') and "epoch" in f]
    if len(model_name) == 0:
        raise FileNotFoundError(f"No model found under {fold_path}")
    else:
        print(f"Selected model: {str(fold_path)}/{model_name[0]} for fold {cfg.Data.fold}")
    #---->Define Model
    ModelInterface_dict = {'model': cfg.Model,
                            'loss': cfg.Loss,
                            'optimizer': cfg.Optimizer,
                            'data': cfg.Data,
                            'log': fold_path
                            }
    cfg.log_path = fold_path
    model = ModelInterface(**ModelInterface_dict)
    # load model
    new_model = model.load_from_checkpoint(checkpoint_path=fold_path/model_name[0], log = fold_path, cfg = cfg)
    # new_model = model.load_state_dict(torch.load(fold_path/model_name[0])['state_dict'])
    trainer.test(model=new_model, datamodule=dm)
    
if __name__ == '__main__':

    args = make_parse()
    cfg = read_yaml(args.config_path)
    # loop over folds
    start = args.k_start if args.k_start else 0
    stop = args.k_end if args.k_end else args.k
    
    for fold in range(start,stop):
        print(f"Computing fold {fold} of {stop} folds.")
        #---->update
        cfg.config = args.config_path
        cfg.General.gpus = args.gpus
        cfg.General.server = 'test'
        cfg.Data.fold = fold

        #---->main
        main(args.checkpoint_path, cfg)