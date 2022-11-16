import argparse
import h5py as h5
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d

from utils import rgbd_to_RGB_point_cloud

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True)
parser.add_argument('--train_ratio', default=.7,type=float)

args = parser.parse_args()
#root
root = args.root
trainscenepath = os.path.join(root,'train_pbr/')
valscenepath = os.path.join(root,'test/')
scenepathlist = [trainscenepath,valscenepath]
model_path= os.path.join(root,'models/')
kpt_path = os.path.join(root,'kpts')
split_path = os.path.join(root,'split')
train_path = os.path.join(split_path,'train')
val_path = os.path.join(split_path,'val')
test_path = os.path.join(split_path,'test')
if not os.path.exists(os.path.join(root+'gt_voters')):
    os.mkdir(os.path.join(root+'gt_voters'))
if not os.path.exists(os.path.join(root+'kpts')):
    os.mkdir(kpt_path)
if not os.path.exists(split_path):
    os.mkdir(split_path)
    os.mkdir(train_path)
    os.mkdir(val_path)
    os.mkdir(test_path)

#generate keypoint
NoofKPoints=9
for model_name in os.listdir(model_path):
    if os.path.splitext(model_name)[1] == '.ply':
        obj_load = o3d.io.read_point_cloud(os.path.join(model_path,model_name))
        BBox = obj_load.get_oriented_bounding_box()
        bboxcorners=np.asarray((BBox.get_box_points()))
        bboxcorners = bboxcorners*2
        #print(bboxcorners)
        xyz = np.asarray(obj_load.points)
       # print(np.mean(xyz[:,0]))
        keypoints=np.zeros((NoofKPoints,3))
        keypoints[0,0] = np.mean(xyz[:,0])
        keypoints[0,1]=np.mean(xyz[:,1])
        keypoints[0,2]=np.mean(xyz[:,2])
        keypoints[1:9,:]=bboxcorners
        #print(keypoints)
        np.save(kpt_path+"/"+os.path.splitext(model_name)[0]+"_sparse9.npy", keypoints)
trainval = [train_path, val_path]
#generate train gt radii
for scenepath in scenepathlist:
    for cycle in os.listdir(scenepath):
        if int(cycle)>=24:
            #print(cycle)
            #paths
            rgbPath = os.path.join(scenepath,cycle,'rgb')
            maskvisPath = os.path.join(scenepath,cycle,'mask_visib')
            depthPath = os.path.join(scenepath,cycle,'depth')

            #jsons
            with open(os.path.join(scenepath, cycle, 'scene_camera.json'), 'r') as f:
                camParas = json.load(f)
            with open(os.path.join(scenepath, cycle, 'scene_gt.json'),'r') as f:
                gtposes = json.load(f)
            for idx in os.listdir(maskvisPath):
                h5f = h5.File(root+'/gt_voters/'+scenepath.split('/')[-1]+'/'+cycle+'.h5','a')
                if not 'pc/'+idx in h5f.keys() and not 'radii/'+idx in h5f.keys() :
                    print(cycle+'/'+idx)
                    #idx = '000653_000000.png'
                    inst_idx = os.path.splitext(idx)[0][7:]
                    idx = os.path.splitext(idx)[0][:6]

                    #print(idx,obj_idx)

                    #load images
                    depth = np.array(Image.open(os.path.join(depthPath,idx+'.png')))
                    #rgb = np.array(Image.open(os.path.join(rgbPath,idx+'.jpg')))
                    rgb=[]
                    #print(os.path.join(maskvisPath,idx+'_'+obj_idx+'.png'))
                    mask_visib = np.array(Image.open(os.path.join(maskvisPath,idx+'_'+inst_idx+'.png')))

                    #load gt info
                    cam_k = np.array(camParas[str(int(idx))]['cam_K'])
                    depth_scale = camParas[str(int(idx))]['depth_scale']
                    gt = gtposes[str(int(idx))][int(inst_idx)]
                    #h5f = h5.File(root+'/gt_voters/'+cycle+'.h5','a')

                    #generate pc
                    pc = rgbd_to_RGB_point_cloud(cam_k.reshape(3,3),depth*depth_scale,rgb)    
                    #print(pc.shape)
                    if not 'pc/'+idx in h5f.keys():
                        h5f.create_dataset('pc/'+idx, data=pc, compression="gzip", compression_opts=9)
                        #save split
                        if scenepath.split('/')[-1] == 'test':
                            with open(os.path.join(test_path,'obj_'+str(gt['obj_id']) +'.txt'),'a')as f:
                                f.write(cycle+'/'+idx+'\n')
                        else:
                            with open(os.path.join(np.random.choice(trainval,1,p=[.7,.3])[0],'obj_'+str(gt['obj_id']) +'.txt'),'a')as f:
                                f.write(cycle+'/'+idx+'\n')    


                    #if obj is visible
                    if np.sum(mask_visib)!=0:
                        vs,us = depth.nonzero()
                        maskList = mask_visib[vs,us]
                        maskList = np.where(maskList==255, int(gt['obj_id']),0)
                        if int(gt['obj_id'])>30:
                            print(gt)
                        #load kpts
                        kpts = np.load(os.path.join(root,'kpts','obj_'+str(gt['obj_id']).zfill(6)+'_sparse9.npy'))

                        pose = np.zeros((3,4))

                        pose[:,:3], pose[:,3] = np.array(gt['cam_R_m2c']).reshape(3,3), np.array(gt['cam_t_m2c'])
                        #print(pose)
                        trans_kpoints = np.dot(kpts, pose[:, :3].T) + pose[:, 3:].T
                        #save radius values 
                        radialList = np.zeros((pc.shape[0],10))
                        radialList[:,0]=maskList
                        i=1
                        #print(pc)
                        for point in trans_kpoints:
                            radialList[:,i] = ((point[0]-pc[:,0])**2+(point[1]-pc[:,1])**2+(point[2]-pc[:,2])**2)**0.5*radialList[:,0]
                            radialList[:,i] *=10
                            i+=1
                        if 'radii/'+idx in h5f.keys():
                            origin_radii = np.array(h5f['radii/'+idx])
                            radialList = np.where((radialList!=0) & (origin_radii==0), radialList, origin_radii)
                            del h5f['radii/'+idx]
                        h5f.create_dataset('radii/'+idx, data=radialList, compression="gzip", compression_opts=9)
                        h5f.close()
