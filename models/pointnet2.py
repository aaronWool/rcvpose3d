#modified based on https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation
import time

batch_size=4

class Pointnet2(nn.Module):
    def __init__(self, output_channel, input_channel=3):
        super(Pointnet2, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], input_channel, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, output_channel, 1)
        self.sgimoid = nn.Sigmoid()

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        #tik=time.perf_counter()
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        #tok=time.perf_counter()
        #print("encoding time:", (tok-tik),"s")
        #print(l1_xyz.size(),l1_points.size(),l2_xyz.size(),l2_points.size(),l3_xyz.size(),l3_points.size(),l4_xyz.size(),l4_points.size())
        
        #tik=time.perf_counter()
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        #tok=time.perf_counter()
        #print("decoding time: ", (tok-tik), "s")

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = self.sgimoid(x)
        #x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x

class Pointnet2_5(nn.Module):
    def __init__(self, output_channel):
        super(Pointnet2_5, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 3, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 192, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.sa5 = PointNetSetAbstractionMsg(8, [0.8, 1.0], [16, 16], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp5 = PointNetFeaturePropagation(1024+1024+512+512, [512, 512])
        self.fp4 = PointNetFeaturePropagation(512+512, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 64])
        self.conv1 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(64, output_channel, 1)
        self.sgimoid = nn.Sigmoid()

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        #tik=time.perf_counter()
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)
        #tok=time.perf_counter()
        #print("encoding time:", (tok-tik),"s")
        #print(l1_xyz.size(),l1_points.size(),l2_xyz.size(),l2_points.size(),l3_xyz.size(),l3_points.size(),l4_xyz.size(),l4_points.size())
        
        #tik=time.perf_counter()
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        #tok=time.perf_counter()
        #print("decoding time: ", (tok-tik), "s")

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = self.sgimoid(x)
        #x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x

class Pointnet2_6(nn.Module):
    def __init__(self, output_channel):
        super(Pointnet2_6, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.025, 0.05], [16, 32], 3, [[8, 8, 16], [16, 16, 32]])
        self.sa2 = PointNetSetAbstractionMsg(512, [0.05, 0.1], [16, 32], 16+32, [[16, 16, 32], [32, 32, 64]])
        self.sa3 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa4 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 192, 256]])
        self.sa5 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.sa6 = PointNetSetAbstractionMsg(8, [0.8, 1.0], [16, 16], 512+512, [[512, 512, 1024], [512, 768, 1024]])
        self.fp6 = PointNetFeaturePropagation(1024+1024+512+512, [512, 512])
        self.fp5 = PointNetFeaturePropagation(512+512, [256, 256])
        self.fp4 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp2 = PointNetFeaturePropagation(176, [128, 64])
        self.fp1 = PointNetFeaturePropagation(64, [64, 64, 64])
        self.conv1 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(64, num_radius, 1)
        self.sgimoid = nn.Sigmoid()

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        #tik=time.perf_counter()
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points)
        #tok=time.perf_counter()
        #print("encoding time:", (tok-tik),"s")
        #print(l1_xyz.size(),l1_points.size(),l2_xyz.size(),l2_points.size(),l3_xyz.size(),l3_points.size(),l4_xyz.size(),l4_points.size())
        
        #tik=time.perf_counter()
        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        #tok=time.perf_counter()
        #print("decoding time: ", (tok-tik), "s")

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = self.sgimoid(x)
        #x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x

class Pointnet2_7(nn.Module):
    def __init__(self, output_channel, input_channel=3):
        super(Pointnet2_7, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.0125, 0.025], [16, 32], input_channel, [[4, 4, 8], [8, 8, 16]])
        self.sa2 = PointNetSetAbstractionMsg(512, [0.025, 0.05], [16, 32], 8+16, [[8, 8, 16], [16, 16, 32]])
        self.sa3 = PointNetSetAbstractionMsg(512, [0.05, 0.1], [16, 32], 16+32, [[16, 16, 32], [32, 32, 64]])
        self.sa4 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa5 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 192, 256]])
        self.sa6 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.sa7 = PointNetSetAbstractionMsg(8, [0.8, 1.0], [16, 16], 512+512, [[512, 512, 1024], [512, 768, 1024]])
        self.fp7 = PointNetFeaturePropagation(1024+1024+512+512, [512, 512])
        self.fp6 = PointNetFeaturePropagation(512+512, [256, 256])
        self.fp5 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp4 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp3 = PointNetFeaturePropagation(176, [128, 64])
        self.fp2 = PointNetFeaturePropagation(88, [64, 32])
        self.fp1 = PointNetFeaturePropagation(32, [32, 32, 32])
        self.conv1 = nn.Conv1d(32, 16, 1)
        self.bn1 = nn.BatchNorm1d(16)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(16, output_channel, 1)
        self.sgimoid = nn.Sigmoid()

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        #tik=time.perf_counter()
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points)
        l7_xyz, l7_points = self.sa7(l6_xyz, l6_points)
        #tok=time.perf_counter()
        #print("encoding time:", (tok-tik),"s")
        #print(l1_xyz.size(),l1_points.size(),l2_xyz.size(),l2_points.size(),l3_xyz.size(),l3_points.size(),l4_xyz.size(),l4_points.size())
        
        #tik=time.perf_counter()
        l6_points = self.fp7(l6_xyz, l7_xyz, l6_points, l7_points)
        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        #tok=time.perf_counter()
        #print("decoding time: ", (tok-tik), "s")

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = self.sgimoid(x)
        #x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x

if __name__ == '__main__':
    import  torch
    from pytorch_model_summary import summary
    from caspointnet2 import Cas_Pointnet2_7
    input = torch.rand(2,3,32768)
    cropped_input = torch.rand(2,3,1024)
    model = Cas_Pointnet2_7()
    model.train()
    out_sem,out_r = model(input, cropped_input)
    print(out_sem.shape)
    print(out_r.shape)
    model.eval()
    out_r = model(input)
    print(out_r.shape)
    
    
