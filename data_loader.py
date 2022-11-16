# from torchvision.datasets import VOCSegmentation
from Radii3D import Radii3D
from torch.utils import data
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
#		globalmin = 0.41948623669063784 
#        globalmax = 1.5384439645988315

class RadiiData(Radii3D):
    def __init__(self, root, trainval='train', obj_idx='1',globalmin=0,globalmax=1):
        transform = self.transform
        self.obj_idx = obj_idx
        self.globalmin = globalmin
        self.globalmax = globalmax
        super(RadiiData, self).__init__(root,
                                          trainval=trainval,
                                          obj_idx=obj_idx,
                                          transform=transform)

    def transform(self, pc, target):
        pc=pc[:,0:3]
        pc[:,0]-=np.average(pc[:,0])
        pc[:,1]-=np.average(pc[:,1])
        pc[:,2]-=np.average(pc[:,2])
        if(pc.shape[0]<target.shape[0]):
            nP = pc.shape[0]
        else:
            nP = target.shape[0]
        idx=np.random.choice(nP,size=32768,replace=False)
        sem_idx=np.where(target[:,0]==1)
        #print(idx[0])
        if len(sem_idx[0])>1024:
            sem_idx=np.random.choice(sem_idx[0],size=1024,replace=False)
        #else:
        cropped_pc = np.zeros((1024,pc.shape[1]))
        cropped_target = np.zeros((1024,target.shape[1]))

        down_sampled_pc = pc[idx]
        tmp_cropped_pc = pc[sem_idx]
        cropped_pc[:tmp_cropped_pc.shape[0],:]=tmp_cropped_pc
        down_sampled_target = target[idx,0]
        tmp_cropped_target = target[sem_idx]
        cropped_target[:tmp_cropped_target.shape[0],:]=tmp_cropped_target
        #down_sampled_target = down_sampled_target[:,0]
        tmp_array = np.zeros((cropped_target.shape[0],3))
        tmp_array[:,1] = cropped_target[:,3]
        tmp_array[:,2] = cropped_target[:,4]
        tmp_array[:,0] = cropped_target[:,2]
        cropped_target = tmp_array
        #down_sampled_target*=10
        #0.35280348582397203 0.0629499862995941
        cropped_target = np.where(cropped_target!=0, (cropped_target - self.globalmin) / (self.globalmax-self.globalmin),0)
        #down_sampled_target[:,1:4] /=10
        down_sampled_pc=np.transpose(down_sampled_pc)
        cropped_pc = np.transpose(cropped_pc)
        #cropped_pc = cropped_pc.astype(np.float32)
       #pc = pc.astype(np.float32)
        #down_sampled_target = down_sampled_target.astype(np.float32)
        #cropped_target =cropped_target.astype(np.float32)
        down_sampled_pc = torch.from_numpy(down_sampled_pc).float()
        cropped_pc = torch.from_numpy(cropped_pc).float()
        sem_gt = torch.from_numpy(down_sampled_target).float()
        radii_gt = torch.from_numpy(cropped_target).float()
        return down_sampled_pc, cropped_pc, sem_gt, radii_gt 

    def __len__(self):
        return len(self.ids)


def get_loader(opts):
    from data_loader import RadiiData
    import os
    kwargs = {'num_workers': 0} if 'True' in str(opts.cuda) else {}
    obj = opts.obj_idx
    train_loader = data.DataLoader(RadiiData(opts.root_dataset,
                                               trainval='train', 
                                               obj_idx=obj,
                                               globalmin=opts.globalmin,
                                               globalmax=opts.globalmax),
                                   batch_size=int(opts.batch_size),
                                   shuffle=True,
                                   drop_last=True,
                                   **kwargs)
    val_loader = data.DataLoader(RadiiData(opts.root_dataset,
                                             trainval='val', 
                                             obj_idx=obj,
                                             globalmin=opts.globalmin,
                                             globalmax=opts.globalmax),
                                 batch_size=int(opts.batch_size),
                                 shuffle=False,
                                 **kwargs)
    return train_loader, val_loader
