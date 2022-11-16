import torch
import torch.nn as nn

from models.pointnet2 import Pointnet2_7, Pointnet2

class Cas_Pointnet2_7(nn.Module):
    def __init__(self, input_channel=3,sem_threshold=0.7):
        super(Cas_Pointnet2_7, self).__init__()
        self.sem_threshold = sem_threshold
        self.sem = Pointnet2_7(1,input_channel)
        self.regression = Pointnet2(3,input_channel)
            
    def forward(self,input = [], cropped_input = []):
        if torch.is_tensor(cropped_input):
                sem_output = self.sem(input)
                radii_output = self.regression(cropped_input)
                output = [sem_output,radii_output]
        else:
            sem_label = self.sem(input)
            sem_label[:,:1024,:]=1
            sem_label = sem_label[:,:,0]
            #idx = torch.where(sem_label>=self.sem_threshold)
            input = torch.permute(input,(0,2,1))
            xyz_seg = torch.zeros(input.shape[0],1024,input.shape[2])
            xyz_seg = xyz_seg.type_as(input)
            print(xyz_seg.shape)
            for i in range(input.shape[0]):
                idx = torch.where(sem_label[i,]>=self.sem_threshold)
                xyz_seg[idx[0],:,i] = input[idx[0],idx[1],i]
            #print(xyz_seg.shape)
            xyz_seg = torch.permute(xyz_seg,(0,2,1))
            #print(xyz_seg.shape)
            output = self.regression(xyz_seg)
        return output

if __name__ == '__main__':
    import  torch
    #from pytorch_model_summary import summary
    from caspointnet2 import Cas_Pointnet2_7
    input = torch.rand(2,3,32768).cuda()
    cropped_input = torch.rand(2,3,1024).cuda()
    model = Cas_Pointnet2_7()
    model = model.cuda()
    model.train()
    out_sem,out_r = model(input, cropped_input)
    print(out_sem.shape)
    print(out_r.shape)
    model.eval()
    out_r = model(input)
    print(out_r.shape)
