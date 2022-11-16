from itertools import cycle
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import torch
from matplotlib import pyplot
import h5py as h5

class Radii3D(Dataset):
    def __init__(self, root, trainval, obj_idx="1", transform=None):
        self.root = root
        self.trainval = trainval
        self.obj_id = obj_idx
        self.transform = transform
        self._trainvalpath = os.path.join(self.root,'split',trainval, 'obj_'+obj_idx+'.txt')

        with open(self._trainvalpath) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]

    def __getitem__(self, item):
        cycle_scene_id = self.ids[item]
        cycle_id, scene_id = cycle_scene_id.split('/')
        gt_h5 = os.path.join(self.root,'gt_voters',cycle_id+'.h5')
        with h5.File(gt_h5,'r') as f:
            pc = np.array(f['pc/'+scene_id])
            target = np.array(f['radii/'+scene_id])
        if self.transform is not None:
            pc_torch, cropped_pc_torch, sem_torch,radii_torch = self.transform(pc, target)
        return pc_torch, cropped_pc_torch, sem_torch,radii_torch

    def __len__(self):
        return len(self.ids)
