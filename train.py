import os
import torch
import pytorch_lightning as pl
from models.caspointnet2 import Cas_Pointnet2_7
from models.pointnet2 import Pointnet2_7

class Cascade_RCVPose(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        #input channel is 3 if xyz only, 6 if xyz with rgb, 9 if xyz with rgb and normals
        self.model = Cas_Pointnet2_7(input_channel=3, sem_threshold=.7)
        self.args = args
        self.sem_l = torch.nn.BCELoss()
        self.radii_l = torch.nn.SmoothL1Loss()
        self.globalmax = args.globalmax
        self.globalmin = args.globalmin

    def geo_l(self, pred, target):
        '''
        pred shape: [B, N, 3]
        target shape: [B, N, 3]
        '''
        B,N,_ = pred.shape
        diff01  = torch.abs((pred[:, :, 0] - pred[:, :, 1])-(target[:,:,0]-target[:,:,1]))
        diff01 = torch.where(diff01<1, 0.5*torch.square(diff01),diff01-0.5)
        diff02 = torch.abs((pred[:, :, 0] - pred[:, :, 2])-(target[:,:,0]-target[:,:,2]))
        diff02 = torch.where(diff02<1, 0.5*torch.square(diff02),diff02-0.5)
        diff12 = torch.abs((pred[:, :, 1] - pred[:, :, 2])-(target[:,:,1]-target[:,:,2]))
        diff12 = torch.where(diff12<1, 0.5*torch.square(diff12),diff12-0.5)
        loss = torch.sum(diff01+diff02+diff12)/(B*N*3)
        return loss

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        input, cropped_input, sem_gt, radii_gt = batch
        sem_out,radii_out = self.model(input,cropped_input) 
        loss_s = self.sem_l(sem_out[:,:,0],sem_gt)
        loss_sl1r = self.radii_l(radii_out,radii_gt)
        loss_geo = self.geo_l(radii_out,radii_gt)
        #different weight for initial training and fine tuning
        loss_r = 0.8*loss_sl1r + 0.2*loss_geo 
        loss = loss_s+loss_r
        self.log('train_sem', loss_s)
        self.log('train_r',loss_r)
        self.log('train_rsl1',loss_sl1r)
        self.log('train_rgeo',loss_geo)
        self.log('train', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input, cropped_input, sem_gt, radii_gt = batch
        sem_out,radii_out = self.model(input,cropped_input) 
        loss_s = self.sem_l(sem_out[:,:,0],sem_gt)
        #print(radii_gt.shape,radii_out.shape)
        loss_sl1r = self.radii_l(radii_out,radii_gt)
        loss_geo = self.geo_l(radii_out,radii_gt)
        #different weight for initial training and fine tuning
        loss_r = .8*loss_sl1r + .2*loss_geo 
        loss = loss_s+loss_r
        self.log('Val_sem', loss_s)
        self.log('val_r', loss_r)
        self.log('val_rsl1',loss_sl1r)
        self.log('val_geo',loss_geo)
        self.log('val_loss', loss)
        for i in range(3):
            'Val_ACC_kp'+str(i), float(torch.sum(torch.where(torch.abs(radii_out[:,:,i]-radii_gt[:,:,i])[torch.where(radii_gt[:,:,i]!=0)]*(self.globalmax-self.globalmin)<=0.5,1,0)) / float(len(torch.nonzero(radii_gt[:,:,i]))))
        #self.log('train_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.args.lr)
        return optimizer
