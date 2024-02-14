import torch
from data_loader import get_loader
from train import Cascade_RCVPose
import warnings
from tensorboardX import SummaryWriter
import os
import pytorch_lightning as pl
warnings.filterwarnings('ignore')
resume = ''

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # Parameters to set
    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        choices=['train', 'test'])
    parser.add_argument("--root_dataset",
                        type=str,
                        required=True,
                        default='dataset/LINEMOD/')
    parser.add_argument("--obj_idx",
                        type=str,
                        required=True,
                        default='1')
    parser.add_argument("--resume", type=str, default='')
    parser.add_argument("--optimizer",
                        type=str,
                        default='Adam',
                        choices=['Adam', 'SGD'])
    parser.add_argument("--lr",
                        type=float,
                        default=1e-3)                    
    parser.add_argument("--batch_size",
                        type=str,
                        default='2')
    parser.add_argument("--log_dir",
                        type=str,
                        default='logs/')
    parser.add_argument("--globalmin",
                        type=float)
    parser.add_argument("--globalmax",
                        type=float)
    parser.add_argument("--cuda",
                        type=str,
                        default='True',
                        choices={'True','False'})
    parser.add_argument("--train_status",
                        type=str,
                        default='sem',
                        choices={'sem','radii'})
    args = parser.parse_args()
    
    #check if gou available
    if torch.cuda.is_available():
        noofgpus=torch.cuda.device_count()
        device = 'gpu'
    else:
        device = 'cpu'
    #check no of gpus if so
    model = Cascade_RCVPose(args)
    train_loader, val_loader = get_loader(args)
    trainer = pl.Trainer(limit_train_batches=1.0, max_epochs=500,devices=noofgpus,
    accelerator=device)
    trainer.fit(model=model, train_dataloaders=train_loader,val_dataloaders=val_loader)

