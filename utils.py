import numpy as np


def rgbd_to_RGB_point_cloud(K, depth,img=[]):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    if img !=[]:
        r = img[vs,us,0]
        g = img[vs,us,1]
        b = img[vs,us,2]
    #print(zs)
    #print(zs.max())
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    if img ==[]:
        pts = np.array([xs,ys,zs]).T
    else:
        pts = np.array([xs, ys, zs,r/255,g/255,b/255]).T
    return pts