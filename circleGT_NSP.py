import cv2,os
import pandas as pd
import numpy as np

def circle_mark(im_name = "",star_path='./png/StarFiles/',ostar_path='./mrc/StarFiles/'):
    dstn='%s/check/'%(star_path)
    try:
         os.mkdir(dstn)
    except:
            a=0
    im = cv2.imread(im_name)
    strfl='%s%s.star'%(star_path,'.'.join(im_name.split('.')[0:-1]).split('/')[-1])
    #print(strfl)
    gt_df = pd.read_csv(strfl, skiprows=11, header=None,sep='\s+')

    falcon_cord=gt_df[[0,1]]
    falcon_tuples = [tuple(x) for x in falcon_cord.values]
    for num in falcon_tuples:
        cv2.circle(im,(int(num[0]),int((num[1]))),50,(0,255,0),3)
    try:
        strfl='%s%s.star'%(ostar_path,'.'.join(im_name.split('.')[0:-1]).split('/')[-1])
        gt_df = pd.read_csv(strfl, skiprows=11, header=None,sep='\s+')

        falcon_cord=gt_df[[0,1]]
        falcon_tuples = [tuple(x) for x in falcon_cord.values]
        for num in falcon_tuples:
          cv2.circle(im,(int(num[0]),int((num[1]))),47,(255,255,255),3)
    except:
        print("No Ground Truth starfile for %s"%im_name)
       
    fl='./%s_star.png'%im_name.split('.')[0:-1][1].split('/')[-1]
    print('Saving File: %s'%fl)
    cv2.imwrite(dstn+fl, im)
    return(im)

def getlist(path='./png/StarFiles/selected/'):
    from os import listdir
    imlist = []
    x = np.unique(([i for i in listdir(path)]))
    for i in x:
        fl=i.split('.')[0:-1][0]
        if os.path.isfile("%s/%s.png" %(path,fl)):
            imlist.append("%s/%s.png" %(path,fl))
    return(imlist)

imgs=getlist()
for i in imgs:
    im=circle_mark(i)
