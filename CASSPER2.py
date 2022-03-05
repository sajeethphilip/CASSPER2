import os,sys
try:
    import matplotlib
except:
    print('Please run pip install -r requirements.txt first')

    sys.exit()
import warnings

warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from PIL import ImageEnhance
from scipy import ndimage
from scipy import misc
try:
    from scipy.misc import imsave
except:
    from keras.preprocessing.image import save_img as imsave
import scipy.misc
import scipy
import image_slicer
from image_slicer import join
import glob
from skimage import data, img_as_float
from skimage import exposure
import time, math
import tensorflow as tf
import argparse
import csv

import subprocess

#from tensorflow.contrib import slim
import tf_slim as slim
import os,shutil,sys,cv2,pickle
import numpy as np

import os,time,cv2, sys, math
#import tensorflow.contrib.slim as slim
import time, datetime
import argparse
import random
import os, sys
import subprocess
try:

    from scipy.misc import imread
except:
    from imageio import imread

import ast
Window_Size=512
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

from shutil import copy

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


try:
        os.mkir('models')
except:
        a=0

import tensorflow as tf
#from tensorflow.contrib import slim
tf.compat.v1.disable_eager_execution()

def Upsampling(inputs,scale):
    return tf.image.resize(inputs, size=[tf.shape(input=inputs)[1]*scale,  tf.shape(input=inputs)[2]*scale], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

def Unpooling(inputs,scale):
    return tf.image.resize(inputs, size=[tf.shape(input=inputs)[1]*scale,  tf.shape(input=inputs)[2]*scale], method=tf.image.ResizeMethod.BILINEAR)

def ResidualUnit(inputs, n_filters=48, filter_size=3):
    """
    A local residual unit

    Arguments:
      inputs: The input tensor
      n_filters: Number of output feature maps for each conv
      filter_size: Size of convolution kernel

    Returns:
      Output of local residual block
    """

    net = slim.conv2d(inputs, n_filters, filter_size, activation_fn=None)
    net = slim.batch_norm(net, fused=True)
    net = tf.nn.relu(net)
    net = slim.conv2d(net, n_filters, filter_size, activation_fn=None)
    net = slim.batch_norm(net, fused=True)

    return net

def Check_results():
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
   

def get_radius_erode(image=None,shimg=None,Train=False):
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    # create trackbars
    
    cv2.createTrackbar('radius','image',60,500,callback)
    cv2.createTrackbar('erosion','image',200,1000, callback)
    cv2.createTrackbar('dilation','image',400,1000, callback)
    cv2.createTrackbar('Kernel','image',2,200, callback)
    cv2.createTrackbar('Weight','image',200,200, callback)
    cv2.createTrackbar('threshold','image',1,255, callback)    
    oldradius=60
    olderode=200
    olddial=400
    oldkvl=2
    oldwt=200
    oldtp=1
    radss=[]
    eros=[]
    dials=[]
    kvals=[]
    wvals=[]
    thresh=[]
    oldthresh=128
    print('Reading Image:%s'%image)
    imx = cv2.imread(image)
    sx,sy,_=imx.shape
    shimx = cv2.imread(shimg)

    img=imx[sx//2-Window_Size:sx//2+Window_Size,sy//2-Window_Size:sy//2+Window_Size]
    sximg=shimx[sx//2-Window_Size:sx//2+Window_Size,sy//2-Window_Size:sy//2+Window_Size]
    oldx=sx//2
    oldy=sy//2
    PCCode=np.array([label_values[class_names_list=='Protein']])
    print("---------------------------------------------------------")
    print("    ------------Verify the selection-----------")
    print("Move frame up,down,left or right by pressing u,d,r,l")
    print("  Adjust radius and other parameters as required")
    print("         Select kernel type by pressing k")
    print("   To save selection press s after each setting")
    print("                  Press q to quit")
    print("---------------------------------------------------------")
    while(1):

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('s'):
                radss.append(radius)
                eros.append(erode)
                dials.append(dial)
                kvals.append(kvl)
                wvals.append(wt)
                thresh.append(oldthresh)
                lastk='s'
            if k== ord('k'):
                if oldtp<2:
                    oldtp=oldtp+1
                else:
                    oldtp=0
                print("Kernel changed to %s"%oldtp)
            if k== ord('u'):
                if oldy>=2*Window_Size:
                    oldy=oldy-Window_Size
                    img=imx[oldx-Window_Size:oldx+Window_Size,oldy-Window_Size:oldy+Window_Size]
                    sximg=shimx[oldx-Window_Size:oldx+Window_Size,oldy-Window_Size:oldy+Window_Size]
                else:
                    oldy=2*Window_Size
            if k== ord('d'):
                if oldy<=sy+2*Window_Size:
                    oldy=oldy+Window_Size
                    img=imx[oldx-Window_Size:oldx+Window_Size,oldy-Window_Size:oldy+Window_Size]
                    sximg=shimx[oldx-Window_Size:oldx+Window_Size,oldy-Window_Size:oldy+Window_Size]
                else:
                    oldy=sy-2*Window_Size
            if k== ord('r'):
                if oldx<=sx+2*Window_Size:
                    oldx=oldx+Window_Size
                    img=imx[oldx-Window_Size:oldx+Window_Size,oldy-Window_Size:oldy+Window_Size]
                    sximg=shimx[oldx-Window_Size:oldx+Window_Size,oldy-Window_Size:oldy+Window_Size]
                else:
                    oldx=sx-2*Window_Size
            if k== ord('l'):
                if oldx>=2*Window_Size:
                    oldx=oldx-Window_Size
                    img=imx[oldx-Window_Size:oldx+Window_Size,oldy-Window_Size:oldy+Window_Size]
                    sximg=shimx[oldx-Window_Size:oldx+Window_Size,oldy-Window_Size:oldy+Window_Size]

                else:
                    oldx=2*Window_Size

            # get current positions of the trackbars
            radius = cv2.getTrackbarPos('radius','image')
            erode = cv2.getTrackbarPos('erosion','image')//10
            dial = cv2.getTrackbarPos('dialation','image')//10
            kvl=cv2.getTrackbarPos('Kernel','image')//20
            wt=cv2.getTrackbarPos('Weight','image')//20
            oldthresh=cv2.getTrackbarPos('threshold','image')//10
            if kvl<1:
                kvl=1
            if wt <1:
                wt=1
            if radius !=oldradius:
                oldradius=radius
                lastk='r'
            if erode !=olderode:
                olderode=erode
                lastk='e'
            if dial !=olddial:
                olddial=dial
                lastk='d'
            if kvl !=oldkvl:
                oldkvl=kvl
                lastk='k'
            if wt !=oldwt:
                oldwt=wt
                lastk='r'
            contr_min=radius*radius*np.pi/wt        #Fill at is 1/8th area
            img[np.any(img != np.flip(PCCode[0],0), axis=-1)]=[0,0,0]
            try:
                gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sxgray_frame = cv2.cvtColor(sximg, cv2.COLOR_BGR2GRAY)
            except:
                continue
            if oldtp==0:
                  kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(kvl,kvl))
            if oldtp==1:
                  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kvl,kvl))
            if oldtp==2:
                 kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kvl,kvl))
            thresh01 = np.uint8(cv2.erode(gray_frame, kernel,iterations=erode))
            thresh01 = cv2.dilate(thresh01,kernel,iterations=dial)
            thresh12 = cv2.distanceTransform(thresh01,cv2.DIST_L2,3)
            ret,thresh1 = cv2.threshold(thresh12,oldthresh,255,cv2.THRESH_BINARY)
            # The next is a dummy operation
            thresh1 = np.uint8(cv2.erode(thresh1, kernel,iterations=1))
 
            #circle_frame=cv2.cvtColor(thresh1,cv2.COLOR_GRAY2BGR)
            circle_frame=cv2.cvtColor(sxgray_frame,cv2.COLOR_GRAY2BGR)
            try:
                contours,h= cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            except:
                _,contours,h= cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            mcontours=np.median([cv2.contourArea(c) for c in contours])
            for c in contours:
                if cv2.contourArea(c)>contr_min and cv2.contourArea(c)<4*wt*contr_min:
                    (x,y),_ = cv2.minEnclosingCircle(c)
                    center = (int(x),int(y))
                    cv2.circle(circle_frame, center, radius, (0, 255, 0), 3)
            if k == ord('s') or lastk=='s':
                for c in contours:
                    if cv2.contourArea(c)>contr_min and cv2.contourArea(c)<4*wt*contr_min:
                        (x,y),_ = cv2.minEnclosingCircle(c)
                        center = (int(x),int(y))
                        cv2.circle(sximg, center, 2, (255, 255, 255), 4)

                    
            cv2.imshow('image',circle_frame)
    radss.append(radius)
    eros.append(erode)
    dials.append(dial)
    kvals.append(kvl)
    wvals.append(wt)
    thresh.append(oldthresh)
    print("Your required Radius are",radss)
    print("Your required erosion count are",eros)
    print("Your required dilation count are",dials)
    print("Your required kernels are ",kvals)
    print("Your required weights are ",wvals)
    cv2.destroyAllWindows()
    return(radss,eros,dials,kvals,wvals,thresh)


def FullResolutionResidualUnit(pool_stream, res_stream, n_filters_3, n_filters_1, pool_scale):
    """
    A full resolution residual unit

    Arguments:
      pool_stream: The inputs from the pooling stream
      res_stream: The inputs from the residual stream
      n_filters_3: Number of output feature maps for each 3x3 conv
      n_filters_1: Number of output feature maps for each 1x1 conv
      pool_scale: scale of the pooling layer i.e window size and stride

    Returns:
      Output of full resolution residual block
    """

    G = tf.concat([pool_stream, slim.pool(res_stream, [pool_scale, pool_scale], stride=[pool_scale, pool_scale], pooling_type='MAX')], axis=-1)



    net = slim.conv2d(G, n_filters_3, kernel_size=3, activation_fn=None)
    net = slim.batch_norm(net, fused=True)
    net = tf.nn.relu(net)
    net = slim.conv2d(net, n_filters_3, kernel_size=3, activation_fn=None)
    net = slim.batch_norm(net, fused=True)
    pool_stream_out = tf.nn.relu(net)

    net = slim.conv2d(pool_stream_out, n_filters_1, kernel_size=1, activation_fn=None)
    net = Upsampling(net, scale=pool_scale)
    res_stream_out = tf.add(res_stream, net)

    return pool_stream_out, res_stream_out



def build_frrn(inputs, num_classes, preset_model='FRRN-A'):
    """
    Builds the Full Resolution Residual Network model.

    Arguments:
      inputs: The input tensor
      preset_model: Which model you want to use. Select FRRN-A or FRRN-B
      num_classes: Number of classes

    Returns:
      FRRN model
    """

    if preset_model == 'FRRN-A':

        #####################
        # Initial Stage
        #####################
        net = slim.conv2d(inputs, 48, kernel_size=5, activation_fn=None)
        net = slim.batch_norm(net, fused=True)
        net = tf.nn.relu(net)

        net = ResidualUnit(net, n_filters=48, filter_size=3)
        net = ResidualUnit(net, n_filters=48, filter_size=3)
        net = ResidualUnit(net, n_filters=48, filter_size=3)


        #####################
        # Downsampling Path
        #####################
        pool_stream = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
        res_stream = slim.conv2d(net, 32, kernel_size=1, activation_fn=None)

        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=96, n_filters_1=32, pool_scale=2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=96, n_filters_1=32, pool_scale=2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=96, n_filters_1=32, pool_scale=2)

        pool_stream = slim.pool(pool_stream, [2, 2], stride=[2, 2], pooling_type='MAX')
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=4)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=4)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=4)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=4)

        pool_stream = slim.pool(pool_stream, [2, 2], stride=[2, 2], pooling_type='MAX')
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=384, n_filters_1=32, pool_scale=8)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=384, n_filters_1=32, pool_scale=8)

        pool_stream = slim.pool(pool_stream, [2, 2], stride=[2, 2], pooling_type='MAX')
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=384, n_filters_1=32, pool_scale=16)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=384, n_filters_1=32, pool_scale=16)

        #####################
        # Upsampling Path
        #####################
        pool_stream = Unpooling(pool_stream, 2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=8)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=8)

        pool_stream = Unpooling(pool_stream, 2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=4)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=4)

        pool_stream = Unpooling(pool_stream, 2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=96, n_filters_1=32, pool_scale=2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=96, n_filters_1=32, pool_scale=2)

        pool_stream = Unpooling(pool_stream, 2)

        #####################
        # Final Stage
        #####################
        net = tf.concat([pool_stream, res_stream], axis=-1)
        net = ResidualUnit(net, n_filters=48, filter_size=3)
        net = ResidualUnit(net, n_filters=48, filter_size=3)
        net = ResidualUnit(net, n_filters=48, filter_size=3)

        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
        return net


    elif preset_model == 'FRRN-B':
        #####################
        # Initial Stage
        #####################
        net = slim.conv2d(inputs, 48, kernel_size=5, activation_fn=None)
        net = slim.batch_norm(net, fused=True)
        net = tf.nn.relu(net)

        net = ResidualUnit(net, n_filters=48, filter_size=3)
        net = ResidualUnit(net, n_filters=48, filter_size=3)
        net = ResidualUnit(net, n_filters=48, filter_size=3)


        #####################
        # Downsampling Path
        #####################
        pool_stream = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
        res_stream = slim.conv2d(net, 32, kernel_size=1, activation_fn=None)

        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=96, n_filters_1=32, pool_scale=2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=96, n_filters_1=32, pool_scale=2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=96, n_filters_1=32, pool_scale=2)

        pool_stream = slim.pool(pool_stream, [2, 2], stride=[2, 2], pooling_type='MAX')
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=4)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=4)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=4)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=4)

        pool_stream = slim.pool(pool_stream, [2, 2], stride=[2, 2], pooling_type='MAX')
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=384, n_filters_1=32, pool_scale=8)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=384, n_filters_1=32, pool_scale=8)

        pool_stream = slim.pool(pool_stream, [2, 2], stride=[2, 2], pooling_type='MAX')
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=384, n_filters_1=32, pool_scale=16)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=384, n_filters_1=32, pool_scale=16)

        pool_stream = slim.pool(pool_stream, [2, 2], stride=[2, 2], pooling_type='MAX')
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=384, n_filters_1=32, pool_scale=32)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=384, n_filters_1=32, pool_scale=32)

        #####################
        # Upsampling Path
        #####################
        pool_stream = Unpooling(pool_stream, 2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=16)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=16)

        pool_stream = Unpooling(pool_stream, 2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=8)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=8)

        pool_stream = Unpooling(pool_stream, 2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=4)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=192, n_filters_1=32, pool_scale=4)

        pool_stream = Unpooling(pool_stream, 2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=96, n_filters_1=32, pool_scale=2)
        pool_stream, res_stream = FullResolutionResidualUnit(pool_stream=pool_stream, res_stream=res_stream, n_filters_3=96, n_filters_1=32, pool_scale=2)

        pool_stream = Unpooling(pool_stream, 2)

        #####################
        # Final Stage
        #####################
        net = tf.concat([pool_stream, res_stream], axis=-1)
        net = ResidualUnit(net, n_filters=48, filter_size=3)
        net = ResidualUnit(net, n_filters=48, filter_size=3)
        net = ResidualUnit(net, n_filters=48, filter_size=3)

        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
        return net

    else:
        raise ValueError("Unsupported FRRN model '%s'. This function only supports FRRN-A and FRRN-B" % (preset_model))


# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains common code shared by all inception models.

Usage of arg scope:
  with slim.arg_scope(inception_arg_scope()):
    logits, end_points = inception.inception_v3(images, num_classes,
                                                is_training=is_training)

"""

#import tensorflow as tf

#slim = tf.contrib.slim


def inception_arg_scope(weight_decay=0.00004,
                        use_batch_norm=True,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001,
                        activation_fn=tf.nn.relu,
                        batch_norm_updates_collections=tf.compat.v1.GraphKeys.UPDATE_OPS):
  """Defines the default arg scope for inception models.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    use_batch_norm: "If `True`, batch_norm is applied after each convolution.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.
    activation_fn: Activation function for conv2d.
    batch_norm_updates_collections: Collection for the update ops for
      batch norm.

  Returns:
    An `arg_scope` to use for the inception models.
  """
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      # collection containing update_ops.
      'updates_collections': batch_norm_updates_collections,
      # use fused batch norm if possible.
      'fused': None,
  }
  if use_batch_norm:
    normalizer_fn = slim.batch_norm
    normalizer_params = batch_norm_params
  else:
    normalizer_fn = None
    normalizer_params = {}
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=tf.keras.regularizers.l2(0.5 * (weight_decay))):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=2.0),
        activation_fn=activation_fn,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params) as sc:
      return sc

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the definition of the Inception V4 architecture.

As described in http://arxiv.org/abs/1602.07261.

  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""

#import tensorflow as tf

#slim = tf.contrib.slim


def block_inception_a(inputs, scope=None, reuse=None):
  """Builds Inception-A block for Inception v4 network."""
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.compat.v1.variable_scope(scope, 'BlockInceptionA', [inputs], reuse=reuse):
      with tf.compat.v1.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(inputs, 96, [1, 1], scope='Conv2d_0a_1x1')
      with tf.compat.v1.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(inputs, 64, [1, 1], scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
      with tf.compat.v1.variable_scope('Branch_2'):
        branch_2 = slim.conv2d(inputs, 64, [1, 1], scope='Conv2d_0a_1x1')
        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
      with tf.compat.v1.variable_scope('Branch_3'):
        branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
        branch_3 = slim.conv2d(branch_3, 96, [1, 1], scope='Conv2d_0b_1x1')
      return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])


def block_reduction_a(inputs, scope=None, reuse=None):
  """Builds Reduction-A block for Inception v4 network."""
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.compat.v1.variable_scope(scope, 'BlockReductionA', [inputs], reuse=reuse):
      with tf.compat.v1.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(inputs, 384, [3, 3], stride=2, padding='VALID',
                               scope='Conv2d_1a_3x3')
      with tf.compat.v1.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
        branch_1 = slim.conv2d(branch_1, 256, [3, 3], stride=2,
                               padding='VALID', scope='Conv2d_1a_3x3')
      with tf.compat.v1.variable_scope('Branch_2'):
        branch_2 = slim.max_pool2d(inputs, [3, 3], stride=2, padding='VALID',
                                   scope='MaxPool_1a_3x3')
      return tf.concat(axis=3, values=[branch_0, branch_1, branch_2])


def block_inception_b(inputs, scope=None, reuse=None):
  """Builds Inception-B block for Inception v4 network."""
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.compat.v1.variable_scope(scope, 'BlockInceptionB', [inputs], reuse=reuse):
      with tf.compat.v1.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_1x1')
      with tf.compat.v1.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(branch_1, 224, [1, 7], scope='Conv2d_0b_1x7')
        branch_1 = slim.conv2d(branch_1, 256, [7, 1], scope='Conv2d_0c_7x1')
      with tf.compat.v1.variable_scope('Branch_2'):
        branch_2 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
        branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0b_7x1')
        branch_2 = slim.conv2d(branch_2, 224, [1, 7], scope='Conv2d_0c_1x7')
        branch_2 = slim.conv2d(branch_2, 224, [7, 1], scope='Conv2d_0d_7x1')
        branch_2 = slim.conv2d(branch_2, 256, [1, 7], scope='Conv2d_0e_1x7')
      with tf.compat.v1.variable_scope('Branch_3'):
        branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
        branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
      return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])


def block_reduction_b(inputs, scope=None, reuse=None):
  """Builds Reduction-B block for Inception v4 network."""
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.compat.v1.variable_scope(scope, 'BlockReductionB', [inputs], reuse=reuse):
      with tf.compat.v1.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
        branch_0 = slim.conv2d(branch_0, 192, [3, 3], stride=2,
                               padding='VALID', scope='Conv2d_1a_3x3')
      with tf.compat.v1.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(inputs, 256, [1, 1], scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(branch_1, 256, [1, 7], scope='Conv2d_0b_1x7')
        branch_1 = slim.conv2d(branch_1, 320, [7, 1], scope='Conv2d_0c_7x1')
        branch_1 = slim.conv2d(branch_1, 320, [3, 3], stride=2,
                               padding='VALID', scope='Conv2d_1a_3x3')
      with tf.compat.v1.variable_scope('Branch_2'):
        branch_2 = slim.max_pool2d(inputs, [3, 3], stride=2, padding='VALID',
                                   scope='MaxPool_1a_3x3')
      return tf.concat(axis=3, values=[branch_0, branch_1, branch_2])


def block_inception_c(inputs, scope=None, reuse=None):
  """Builds Inception-C block for Inception v4 network."""
  # By default use stride=1 and SAME padding
  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.compat.v1.variable_scope(scope, 'BlockInceptionC', [inputs], reuse=reuse):
      with tf.compat.v1.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(inputs, 256, [1, 1], scope='Conv2d_0a_1x1')
      with tf.compat.v1.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_1x1')
        branch_1 = tf.concat(axis=3, values=[
            slim.conv2d(branch_1, 256, [1, 3], scope='Conv2d_0b_1x3'),
            slim.conv2d(branch_1, 256, [3, 1], scope='Conv2d_0c_3x1')])
      with tf.compat.v1.variable_scope('Branch_2'):
        branch_2 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_1x1')
        branch_2 = slim.conv2d(branch_2, 448, [3, 1], scope='Conv2d_0b_3x1')
        branch_2 = slim.conv2d(branch_2, 512, [1, 3], scope='Conv2d_0c_1x3')
        branch_2 = tf.concat(axis=3, values=[
            slim.conv2d(branch_2, 256, [1, 3], scope='Conv2d_0d_1x3'),
            slim.conv2d(branch_2, 256, [3, 1], scope='Conv2d_0e_3x1')])
      with tf.compat.v1.variable_scope('Branch_3'):
        branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
        branch_3 = slim.conv2d(branch_3, 256, [1, 1], scope='Conv2d_0b_1x1')
      return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])


def inception_v4_base(inputs, final_endpoint='Mixed_7d', scope=None):
  """Creates the Inception V4 network up to the given final endpoint.

  Args:
    inputs: a 4-D tensor of size [batch_size, height, width, 3].
    final_endpoint: specifies the endpoint to construct the network up to.
      It can be one of [ 'Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
      'Mixed_3a', 'Mixed_4a', 'Mixed_5a', 'Mixed_5b', 'Mixed_5c', 'Mixed_5d',
      'Mixed_5e', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e',
      'Mixed_6f', 'Mixed_6g', 'Mixed_6h', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c',
      'Mixed_7d']
    scope: Optional variable_scope.

  Returns:
    logits: the logits outputs of the model.
    end_points: the set of end_points from the inception model.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
  """
  end_points = {}

  def add_and_check_final(name, net):
    end_points[name] = net
    return name == final_endpoint

  with tf.compat.v1.variable_scope(scope, 'InceptionV4', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):
      # 299 x 299 x 3
      net = slim.conv2d(inputs, 32, [3, 3], stride=2,
                        padding='VALID', scope='Conv2d_1a_3x3')

      end_points["pool1"] = net

      if add_and_check_final('Conv2d_1a_3x3', net): return net, end_points
      # 149 x 149 x 32
      net = slim.conv2d(net, 32, [3, 3], padding='VALID',
                        scope='Conv2d_2a_3x3')
      if add_and_check_final('Conv2d_2a_3x3', net): return net, end_points
      # 147 x 147 x 32
      net = slim.conv2d(net, 64, [3, 3], scope='Conv2d_2b_3x3')
      if add_and_check_final('Conv2d_2b_3x3', net): return net, end_points
      # 147 x 147 x 64
      with tf.compat.v1.variable_scope('Mixed_3a'):
        with tf.compat.v1.variable_scope('Branch_0'):
          branch_0 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_0a_3x3')
        with tf.compat.v1.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 96, [3, 3], stride=2, padding='VALID',
                                 scope='Conv2d_0a_3x3')
        net = tf.concat(axis=3, values=[branch_0, branch_1])

        end_points["pool2"] = net

        if add_and_check_final('Mixed_3a', net): return net, end_points

      # 73 x 73 x 160
      with tf.compat.v1.variable_scope('Mixed_4a'):
        with tf.compat.v1.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_0 = slim.conv2d(branch_0, 96, [3, 3], padding='VALID',
                                 scope='Conv2d_1a_3x3')
        with tf.compat.v1.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 64, [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, 64, [7, 1], scope='Conv2d_0c_7x1')
          branch_1 = slim.conv2d(branch_1, 96, [3, 3], padding='VALID',
                                 scope='Conv2d_1a_3x3')
        net = tf.concat(axis=3, values=[branch_0, branch_1])
        if add_and_check_final('Mixed_4a', net): return net, end_points

      # 71 x 71 x 192
      with tf.compat.v1.variable_scope('Mixed_5a'):
        with tf.compat.v1.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [3, 3], stride=2, padding='VALID',
                                 scope='Conv2d_1a_3x3')
        with tf.compat.v1.variable_scope('Branch_1'):
          branch_1 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
        net = tf.concat(axis=3, values=[branch_0, branch_1])

        end_points["pool3"] = net

        if add_and_check_final('Mixed_5a', net): return net, end_points



      # 35 x 35 x 384
      # 4 x Inception-A blocks
      for idx in range(4):
        block_scope = 'Mixed_5' + chr(ord('b') + idx)
        net = block_inception_a(net, block_scope)
        if add_and_check_final(block_scope, net): return net, end_points

      # 35 x 35 x 384
      # Reduction-A block
      net = block_reduction_a(net, 'Mixed_6a')

      end_points["pool4"] = net

      if add_and_check_final('Mixed_6a', net): return net, end_points

      # 17 x 17 x 1024
      # 7 x Inception-B blocks
      for idx in range(7):
        block_scope = 'Mixed_6' + chr(ord('b') + idx)
        net = block_inception_b(net, block_scope)
        if add_and_check_final(block_scope, net): return net, end_points

      # 17 x 17 x 1024
      # Reduction-B block
      net = block_reduction_b(net, 'Mixed_7a')

      end_points["pool5"] = net

      if add_and_check_final('Mixed_7a', net): return net, end_points

      # 8 x 8 x 1536
      # 3 x Inception-C blocks
      for idx in range(3):
        block_scope = 'Mixed_7' + chr(ord('b') + idx)
        net = block_inception_c(net, block_scope)
        if add_and_check_final(block_scope, net): return net, end_points
  raise ValueError('Unknown final endpoint %s' % final_endpoint)


def inception_v4(inputs, num_classes=1001, is_training=True,
                 dropout_keep_prob=0.8,
                 reuse=None,
                 scope='InceptionV4',
                 create_aux_logits=True):
  """Creates the Inception V4 model.

  Args:
    inputs: a 4-D tensor of size [batch_size, height, width, 3].
    num_classes: number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer (before dropout)
      are returned instead.
    is_training: whether is training or not.
    dropout_keep_prob: float, the fraction to keep before final layer.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    create_aux_logits: Whether to include the auxiliary logits.

  Returns:
    net: a Tensor with the logits (pre-softmax activations) if num_classes
      is a non-zero integer, or the non-dropped input to the logits layer
      if num_classes is 0 or None.
    end_points: the set of end_points from the inception model.
  """
  end_points = {}
  with tf.compat.v1.variable_scope(scope, 'InceptionV4', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = inception_v4_base(inputs, scope=scope)

      # with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
      #                     stride=1, padding='SAME'):
      #   # Auxiliary Head logits
      #   if create_aux_logits and num_classes:
      #     with tf.variable_scope('AuxLogits'):
      #       # 17 x 17 x 1024
      #       aux_logits = end_points['Mixed_6h']
      #       aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3,
      #                                    padding='VALID',
      #                                    scope='AvgPool_1a_5x5')
      #       aux_logits = slim.conv2d(aux_logits, 128, [1, 1],
      #                                scope='Conv2d_1b_1x1')
      #       aux_logits = slim.conv2d(aux_logits, 768,
      #                                aux_logits.get_shape()[1:3],
      #                                padding='VALID', scope='Conv2d_2a')
      #       aux_logits = slim.flatten(aux_logits)
      #       aux_logits = slim.fully_connected(aux_logits, num_classes,
      #                                         activation_fn=None,
      #                                         scope='Aux_logits')
      #       end_points['AuxLogits'] = aux_logits

      #   # Final pooling and prediction
      #   # TODO(sguada,arnoegw): Consider adding a parameter global_pool which
      #   # can be set to False to disable pooling here (as in resnet_*()).
      #   with tf.variable_scope('Logits'):
      #     # 8 x 8 x 1536
      #     kernel_size = net.get_shape()[1:3]
      #     if kernel_size.is_fully_defined():
      #       net = slim.avg_pool2d(net, kernel_size, padding='VALID',
      #                             scope='AvgPool_1a')
      #     else:
      #       net = tf.reduce_mean(net, [1, 2], keep_dims=True,
      #                            name='global_pool')
      #     end_points['global_pool'] = net
      #     if not num_classes:
      #       return net, end_points
      #     # 1 x 1 x 1536
      #     net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b')
      #     net = slim.flatten(net, scope='PreLogitsFlatten')
      #     end_points['PreLogitsFlatten'] = net
      #     # 1536
      #     logits = slim.fully_connected(net, num_classes, activation_fn=None,
      #                                   scope='Logits')
      #     end_points['Logits'] = logits
      #     end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')
    return net, end_points
inception_v4.default_image_size = 299


inception_v4_arg_scope = inception_arg_scope



def scale_write_img(filename="",pth="./",des="",src=""):
        im=((Image.open(src)))
        j,k=pickle.load(open("./Cmetadata/"+filename+'_OrgSz.p', 'rb'))

        #print(j,k)
        #im=cv2.resize(np.asarray(im),(k,j))
        #-----------------------------
        #im=Image.fromarray(im)
        im=Resize_img(im,w2=k,h2=j)
        im.save(pth+des+filename+".png")
        im.close()


#sys.path.append("models")
#from models.FRRN import build_frrn
def circle_mark(im_name = "",star_path='./png/StarFiles/'):
    im = cv2.imread(im_name)
    strfl='%s%s.star'%(star_path,''.join(im_name.split('.')[0:-1]).split('/')[-1])
    print(strfl)
    gt_df = pd.read_csv(strfl, skiprows=11, header=None,sep='\s+')

    falcon_cord=gt_df[[0,1]]
    falcon_tuples = [tuple(x) for x in falcon_cord.values]
    for num in falcon_tuples:
        cv2.circle(im,(int(num[0]),int((num[1]))),50,(0,255,0),7)
    fl='./%s_star.png'%im_name.split('.')[0:-1][1]
    print('Saving File: %s'%fl)
    cv2.imwrite(fl, im)
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


def min_circle(cont):
    contours_poly = cv2.approxPolyDP(cont, 3, True)
    center, _= cv2.minEnclosingCircle(contours_poly)
    return(int(center[0]),int(center[1]))

def min_rect_circle(cont,radius=60):
    contours_poly = cv2.approxPolyDP(cont, 3, True)
    center, _= cv2.minEnclosingCircle(contours_poly)
    rect=cv2.minAreaRect(cont)
    box=np.int0(cv2.boxPoints(rect))
    mn,mx=np.amin(box,axis=0),np.amax(box,axis=0)
    diff=mx-mn
    if np.all(diff<(2*radius+40)):
        return(int(center[0]),int(center[1]))
    else:
        pass

def callback(x):
    pass

def normalize(im):
    max_mrc=np.max(im)
    min_mrc=np.min(im)
    img_original=(254*((im-min_mrc)/(max_mrc-min_mrc))).astype(np.uint8 )
    return(img_original)
def draw_angled_rec(x0, y0, width, height, angle, img):

    _angle = angle * math.pi / 180.0
    b = math.cos(_angle) * 0.5
    a = math.sin(_angle) * 0.5
    pt0 = (int(x0 - a * height - b * width),
           int(y0 + b * height - a * width))
    pt1 = (int(x0 + a * height - b * width),
           int(y0 - b * height - a * width))
    pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
    pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

    cv2.line(img, pt0, pt1, (255, 255, 255), 3)
    cv2.line(img, pt1, pt2, (255, 255, 255), 3)
    cv2.line(img, pt2, pt3, (255, 255, 255), 3)
    cv2.line(img, pt3, pt0, (255, 255, 255), 3)
def draw_all_cntrs(frame=None,img=None):
    im2, contours, hierarchy = cv2.findContours(frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    try:
        hierarchy = hierarchy[0]
    except:
        hierarchy = []

    height, width, _ = img.shape
    min_x, min_y = width, height
    max_x = max_y = 0

    # computes the bounding box for the contour, and draws it on the frame,
    for contour, hier in zip(contours, hierarchy):
        (x,y,w,h) = cv2.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        if w > 80 and h > 80:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)

    if max_x - min_x > 0 and max_y - min_y > 0:
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

def match_star_if_found(file=None,sfdes='mrc/',des='StarFiles/',Box=False):
    #Code by Robin Jacob Roy
    if Box==True:
        sfdex=sfdes+'Box/'
    else:
        sfdex=sfdes+'StarFiles/'
    fl=file.split('/')[-1]
    print(file)
    im = cv2.imread(file)
    if Box==True:
        fln=os.path.join(sfdex,fl.replace('.png','.box'))
        fln2=os.path.join('png/Box/',fl.replace('.png','.box'))
        import pandas as pd
        gt_df2 = pd.read_csv(fln2, skiprows=11, header=None,sep='\s+')
        falcon_cord2=gt_df2[[0,1,2,3,4]]
        falcon_tuples2 = [tuple(x) for x in falcon_cord2.values]
        bx_full=[]
        for num in falcon_tuples2:
            cv2.circle(im,(int(num[0]),int((num[1]))),4,(0,255,0),7)
            xy=((int((num[0])),int((num[1]))))
            hw=((int((num[2])),int((num[3]))))
            ang=((int((num[4]))))
            box=((xy),(hw),(ang))
            bx_full.append(box)
            #box=((((int(num[0])),((int(num[1])))),(((int(num[2]))),((int(num[3])))),((int(num[4])))))
        bx_array = [(np.int0(cv2.boxPoints(c))) for c in bx_full if c is not None]
        for box in bx_array:
                #print(box)
                cv2.drawContours(im,[box] , 0, (0, 255, 0),3)
        try:
            gt_df = pd.read_csv(fln, skiprows=11, header=None,sep='\s+')
            falcon_cord2=gt_df2[[0,1,2,3,4]]
            falcon_tuples2 = [tuple(x) for x in falcon_cord2.values]
            bx_full=[]
            for num in falcon_tuples2:
                cv2.circle(im,(int(num[0]),int((num[1]))),2,(255,255,255),7)
                xy=((int((num[0])),int((num[1]))))
                hw=((int((num[2])),int((num[3]))))
                ang=((int((num[4]))))
                box=((xy),(hw),(ang))
                bx_full.append(box)
                #box=((((int(num[0])),((int(num[1])))),(((int(num[2]))),((int(num[3])))),((int(num[4])))))
            bx_array = [(np.int0(cv2.boxPoints(c))) for c in bx_full if c is not None]
            for box in bx_array:
                #print(box)
                cv2.drawContours(im,[box] , 0, (255, 255, 255),3)
        except:
            print('Failed reading BOX file %s'%(fln) )
            print('If you have a box file with same filename, keep it in')
            print('a subfolder Box/ under the folder that has mrc files')
            print('It will be used to label the png files for validation')
            fln=os.path.join(sfdes+'sf/',fl.replace('.png','.star'))
            try:
                gt_df2 = pd.read_csv(fln, skiprows=11, header=None,sep='\s+')

                falcon_cord2=gt_df2[[0,1]]
                falcon_tuples2 = [tuple(x) for x in falcon_cord2.values]
                for num in falcon_tuples2:
                    cv2.circle(im,(int(num[0]),int((num[1]))),2,(255,255,255),7)
            except:
                a=0

    else:
        fln=os.path.join(sfdes,fl.replace('.png','.star'))
        fln2=os.path.join('png/'+des,fl.replace('.png','.star'))
        try:
            import pandas as pd

            gt_df2 = pd.read_csv(fln2, skiprows=11, header=None,sep='\s+')

            falcon_cord2=gt_df2[[0,1]]
            falcon_tuples2 = [tuple(x) for x in falcon_cord2.values]
            for num in falcon_tuples2:
                cv2.circle(im,(int(num[0]),int((num[1]))),4,(0,255,0),7)
            gt_df = pd.read_csv(fln, skiprows=11, header=None,sep='\s+')

            falcon_cord=gt_df[[0,1]]
            falcon_tuples = [tuple(x) for x in falcon_cord.values]
            for num in falcon_tuples:
                cv2.circle(im,(int(num[0]),int((num[1]))),2,(255,255,255),7)


        except:
            print('Failed reading star file %s'%(fln) )
            print('If you have a star file with same filename, keep it in')
            print('a subfolder sf/ under the folder that has mrc files')
            print('It will be used to label the png files for validation')
            a=0
    cv2.imwrite(file, im)


def getstar(radius=[],erode=[],dial=[],kl=[],wt=[],th=[],image=None,simage=None,des='StarFiles/',mrcsrc="./mrc/",fln='',CrossMatch=False,Box=False):
        imgx=cv2.imread(image)
        simgx=cv2.imread(simage)
        simgx=gray_frame = cv2.cvtColor(simgx, cv2.COLOR_BGR2GRAY)
        simgx=gray_frame = cv2.cvtColor(simgx, cv2.COLOR_GRAY2RGB)
        img=np.copy(imgx)
        PCCode=np.array([label_values[class_names_list=='Protein']])
        final_list=[]
        final_blist=[]
        final_rad=[]
        c2_list=[]
        c3_list=[]
        b2_list=[]
        b3_list=[]
        b_full_list=[]
        for i in range(0, len(radius)):
            rad=radius[i]
            er=erode[i]
            dl=dial[i]
            kval=(kl[i],kl[i])
            wtv=wt[i]
            tho=th[i]
            contr_min=rad*rad*np.pi/wtv                 # 2 means 0.25 % of the area
            #print(contr_min)
            img[np.any(img != np.flip(PCCode[0],0), axis=-1)]=[0,0,0]
            gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,kval)
            thresh01 = cv2.dilate(gray_frame,kernel,iterations=dl)
            thresh01 = np.uint8(cv2.erode(thresh01, kernel,iterations=er))
            thresh12 = cv2.distanceTransform(thresh01,cv2.DIST_L2,3)
            ret,thresh1 = cv2.threshold(thresh12,tho,255,cv2.THRESH_BINARY)
            thresh1 = np.uint8(cv2.erode(thresh1, kernel,iterations=1))
            #circle_frame=cv2.cvtColor(thresh1,cv2.COLOR_GRAY2BGR)
            try:
                contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            except:
                _,contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            cont_array=np.array([c for c in contours])
            cord_ar=[cv2.minAreaRect(c) for c in contours]
            c_ = np.array([cv2.contourArea(contour) for contour in contours])
            #[print("Area is %f"%(int(x)*int(y))) for xx in np.array(cord_ar) for x,y in xx[[1]]]
            [b_full_list.append(tuple(xx)) for xx in np.array(cord_ar) for x,y in xx[[1]] if (((int(x)*int(y))>2*contr_min)&((int(x)*int(y))<=4*wtv*contr_min))]
            bx_array = [(np.int0(cv2.boxPoints(c))) for c in b_full_list if c is not None]
            c_full_list=cont_array[(c_>contr_min) & (c_<=4*wtv*contr_min)]  # 16 means 100% the area
            c_list=(list(map(lambda x: min_rect_circle(x,radius=rad),c_full_list)))
            c_list=[x for x in c_list if x is not None]
            for x in c_list:
                if x not in c2_list:
                    c2_list.append(x)
                    c3_list.append(x)
            final_list.append([rad,[x for x in c3_list if x is not None]])
            c3_list=[]
        if Box==False:
            for r,c in final_list:
                for x,y in c:
                    center = (int(x),int(y))
                    cv2.circle(simgx, center, r, (0, 255, 0), 3)
        else:
            for xx in np.array(b_full_list):
                for x,y in xx[[0]]:
                    #print("box centre=",xx[0])
                    center = (int(x),int(y))
                    cv2.circle(simgx,center, 2, (255, 0, 255),3)

        try:
            os.mkdir('%s/selected/'%des)
        except:
            a=0
        cv2.imwrite('%s/selected/%s'%(des,fln),simgx)

        #print(final_list)
        #final_list=[x for x in c_list if x is not None]
        with open(os.path.join(des,fln.replace('.png','.star')), "w") as starfile:
            starwriter = csv.writer(
            starfile, delimiter="\t", quotechar="|", quoting=csv.QUOTE_NONE)
            starwriter.writerow([])
            starwriter.writerow(["#CASSPER v2.0"])
            #starwriter.writerow([])
            starwriter.writerow(["data_"])
            #starwriter.writerow([])
            starwriter.writerow(["loop_"])
            starwriter.writerow(["_rlnCoordinateX #1 "])
            starwriter.writerow(["_rlnCoordinateY #2"])
            starwriter.writerow(["_rlnRadius #3"])
            starwriter.writerow(["_rlnAnglePsi #4"])
            starwriter.writerow(["_rlnAutopickFigureOfMerit  #5"])
            for r,line in final_list:
                for x,y in line:
                    starwriter.writerow(['{0:.6f}'.format(int(x)),'{0:.6f}'.format(int(y)),'{0:.6f}'.format(int(r)), '{0:.6f}'.format(-999), '{0:.6f}'.format(-999)])

        bdes=os.path.realpath('des/..')
        if not os.path.exists('%s/png/%s/'%(bdes,'Box')):
                os.mkdir('%s/png/%s/'%(bdes,'Box'))
        with open(os.path.join('%s/png/%s/'%(bdes,'Box'),fln.replace('.png','.box')), "w") as boxfile:
            boxwriter = csv.writer(
            boxfile, delimiter="\t", quotechar="|", quoting=csv.QUOTE_NONE)
            boxwriter.writerow([])
            boxwriter.writerow(["#CASSPER v2.0"])
            #starwriter.writerow([])
            boxwriter.writerow(["data_"])
            #starwriter.writerow([])
            boxwriter.writerow(["loop_"])
            boxwriter.writerow(["_rlnCoordinateX #1 "])
            boxwriter.writerow(["_rlnCoordinateY #2"])
            boxwriter.writerow(["_rlnlength#3"])
            boxwriter.writerow(["_rlnbredth #4"])
            boxwriter.writerow(["_rlnAngle  #5"])
            for box in  b_full_list:
                for (x,y),(h,w),c in [box]:
                    boxwriter.writerow(['{0:.6f}'.format(float(x)),'{0:.6f}'.format(float(y)),'{0:.6f}'.format(float(h)), '{0:.6f}'.format(float(w)), '{0:.6f}'.format(float(c))])
        if CrossMatch==True:
            match_star_if_found(file='%s/selected/%s'%(des,fln),sfdes=mrcsrc,Box=Box)
            match_star_if_found(file='png/P_files/%s'%(fln),sfdes=mrcsrc,Box=Box)


SUPPORTED_MODELS = ["FRRN-B"]

SUPPORTED_FRONTENDS = ["InceptionV4"]

def checksetup():
    print('--------------------------------------------------------------------------------')
    print("--Preparing the model. Please wait while I create the required files for you ...")
    print('-----------It may a take a while depending on your internet speed---------------')
    print('--------------------------------------------------------------------------------')
    try:
        os.mkdir('./models/')
    except:
        print('models folder exists')

    if not os.path.isfile("models/CrossPro.ckpt.meta"):
        file_id = '1AuJqYt_NWZCb5Ux_aGzghqZvkUxKiFtk'
        destination = 'models/CrossPro.ckpt.meta'
        download_file_from_google_drive(file_id, destination)
    if not os.path.isfile("models/CrossPro.ckpt.index"):
        file_id = '1Pg5tHySKtgLo2xQKZP6bWfwFAh7lUhwP'
        destination = 'models/CrossPro.ckpt.index'
        download_file_from_google_drive(file_id, destination)
    if not os.path.isfile("models/CrossPro.ckpt.data-00000-of-00001"):
        file_id = '1pwKvdcSWeKTRSp_tkhpTHJpUgaQxyGw2'
        destination = 'models/CrossPro.ckpt.data-00000-of-00001'
        download_file_from_google_drive(file_id, destination)
    if not os.path.isfile("models/class_dict.csv"):
        file_id = '1_6ExBLT9_k8d8w6oYZSNaY0efL0oRR40'
        destination = 'models/class_dict.csv'
        download_file_from_google_drive(file_id, destination)

def download_checkpoints(model_name):
        file_id = '1g8nD1qFe2Z__7dOxvcpTi_liAYHzGv4T'
        destination = 'models/inception_v4.ckpt'
        download_file_from_google_drive(file_id, destination)

def build_model(model_name, net_input, num_classes, crop_width, crop_height, frontend="InceptionV4", is_training=True):
    # Get the selected model.
    # Some of them require pre-trained ResNet
    if "InceptionV4" == frontend and not os.path.isfile("models/inception_v4.ckpt"):
        download_checkpoints("InceptionV4")

    if model_name not in SUPPORTED_MODELS:
        raise ValueError("The model you selected is not supported. The following models are currently supported: {0}".format(SUPPORTED_MODELS))

    if frontend not in SUPPORTED_FRONTENDS:
        raise ValueError("The frontend you selected is not supported. The following models are currently supported: {0}".format(SUPPORTED_FRONTENDS))
    network = None
    init_fn = None
    if model_name == "FRRN-B":
        network = build_frrn(net_input, preset_model = model_name, num_classes=num_classes)
    else:
        raise ValueError("Error: the model %d is not available. Try checking which models are available using the command python main.py --help")

    return network, init_fn


def build_frontend(inputs, frontend, is_training=True, pretrained_dir="models"):
    if frontend == 'InceptionV4':
        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits, end_points = inception_v4.inception_v4(inputs, is_training=is_training, scope='inception_v4')
            frontend_scope='inception_v4'
            init_fn = slim.assign_from_checkpoint_fn(model_path=os.path.join(pretrained_dir, 'inception_v4.ckpt'), var_list=slim.get_model_variables('inception_v4'), ignore_missing_vars=True)
    else:
        raise ValueError("Unsupported fronetnd model '%s'. This function only supports ResNet50, ResNet101, ResNet152, and MobileNetV2" % (frontend))

    return logits, end_points, frontend_scope, init_fn

def tile_image(img):
    w=np.max( np.shape(img))//Window_Size   # we want 512 as max width of the image
    tiles=image_slicer.slice(img, w)
    return(tiles)

def join_image(path,base_img):
    imgs=sorted(glob.glob( os.path.join(path, 'base_img*.png')))
    timage=join(imgs)
    return(timage)


def apply_CLAHE(img,gzw=8,gzh=8,cl=0):
    #-----Splitting the LAB image to different channels-------------------------
    l, a, b=cv2.split(img)
    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe=cv2.createCLAHE(clipLimit=cl, tileGridSize=(gzw,gzh))
    cl=clahe.apply(l)
    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg=cv2.merge((cl,a,b))
    return limg

def apply_CLAHEMono(img,gzw=8,gzh=8,cl=0):
    clahe=cv2.createCLAHE(clipLimit=cl, tileGridSize=(gzw,gzh))
    cl=clahe.apply(img)
    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    return img

def CutMedian(img):
    image1=np.copy(np.asarray(img))
    sigma=np.std(image1)
    mi=np.median(image1)
    mxl=np.max(image1)
    mnl=np.min(image1)
    me=np.mean(image1)
    md=mi - sigma
    mx=mi + sigma
    #print(mxl,mnl,sigma,mx,mi,md,me)
    image1[np.where(image1<md)]=mnl #np.min(image1)
    image1[np.where(image1>mx)]=mxl #np.max(image1)
    image=(254*((image1-mnl)/(mxl-mnl))).astype(np.uint8 )
    #save_mximg(image)
    imag=Image.fromarray(image)
    return(imag)


def CutMedianTH(img):
    image1=np.copy(np.asarray(img))
    sigma=np.std(image1)
    mi=np.median(image1)
    mxl=np.max(image1)
    mnl=np.min(image1)
    me=np.mean(image1)
    md=mi - 0.05*sigma
    mx=mi - 3.5*sigma
    #print(mxl,mnl,sigma,mx,mi,md,me)
    #ret,image1=cv2.threshold(image1,mx,mxl,cv2.THRESH_TOZERO)
    ret,image0=cv2.threshold(image1,mx,mnl,cv2.THRESH_TOZERO)
    ret,image2=cv2.threshold(image1,md,mxl,cv2.THRESH_TOZERO)
    image1=image2-image0
    #image1[np.where(image1<=md)]=mnl #np.min(image1)
    #image1[np.where(image1>=mx)]=mxl #np.max(image1)
    image=(254*((image1-mnl)/(mxl-mnl))).astype(np.uint8)
    save_mximg(image)
    imag=Image.fromarray(image)
    return(imag)

def save_mximg(matrix,filename="test.png"):
    img=Image.fromarray(matrix)
    img.save(filename)
    img.close()

def write_img2(in1,in2,in3,outfl='myimg.png',dest="png/"):
    w,h=np.shape(in1)
    rgbA=np.zeros((w,h,3), 'uint8')
    rgbA[..., 0]=np.copy(in1)
    rgbA[..., 1]=np.copy(in2)
    rgbA[..., 2]=np.copy(in3)
    rgbA=apply_CLAHE(rgbA,gzw=8,gzh=8,cl=0)
    img=Image.fromarray(rgbA)
    #img=Resize_imgAspect(img)
    #img=AdptTh(img)
    img.save(dest+outfl)
    img.close()

def write_img3(in1,in2,in3,outfl='myimg.png',dest="png/"):
    w,h=np.shape(in1)
    in1=apply_CLAHEMono(in1,gzw=512,gzh=512,cl=0)
    imf=ndimage.gaussian_filter(in1,4)
    in1=np.uint8(imf)
    in2=apply_CLAHEMono(in2,gzw=512,gzh=512,cl=0)
    imf=ndimage.gaussian_filter(in2,4)
    in2=np.uint8(imf)
    in3=apply_CLAHEMono(in3,gzw=512,gzh=512,cl=0)
    imf=ndimage.gaussian_filter(in3,4)
    in3=np.uint8(imf)
    rgbA=np.zeros((w,h,3), 'uint8')
    rgbA[..., 0]=np.copy(in1)
    rgbA[..., 1]=np.copy(in2)
    rgbA[..., 2]=np.copy(in3)
    img=Image.fromarray(rgbA)
    #img=Resize_img(img)
    #img=AdptTh(img)
    img.save(dest+outfl)
    img.close()

def AdptTh(image):
    im=np.asarray(image)
    kernel = np.ones((2,2), np.uint8 )
    opening = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img=Image.fromarray(closing.astype(np.uint8 ))
    return img

def write_img(in1,in2,in3,outfl='myimg.png',dest="png/"):
    w,h=np.shape(in1)
    rgbArray = np.zeros((w,h,3), 'uint8')
    rgbArray1 = np.zeros((w,h,1), 'uint8')
    rgbArray2 = np.zeros((w,h,1), 'uint8')
    rgbArray2 = np.copy(in1.astype(np.uint8 ))
    rgbArray[..., 0]=np.copy(rgbArray2)
    img1 = Image.fromarray(rgbArray[..., 0])
    rgbArray2=AdptTh(img1)
    rgbArray[..., 0]=np.asarray(rgbArray2)
    img1 = Image.fromarray(rgbArray[..., 0])
    # image 1 saved as layer 0

    rgbArray1 = np.zeros((w,h,1), 'uint8')
    max_mrc=np.max(in2)
    min_mrc=np.min(in2)

    rgbArray2 = (254*((in2-min_mrc)/(max_mrc-min_mrc))).astype(np.uint8 )
    gzw=w//10
    gzh=h//10
    cl=(0.01*np.mean(rgbArray2))
    rgbArray2=apply_CLAHEMono(rgbArray2,gzw,gzh,cl)
    rgbArray1[..., 0]=np.copy(rgbArray2)
    #rgbArray1[..., 0]=cv2.flip((rgbArray1[..., 0]),0)
    img2 = Image.fromarray(rgbArray1[..., 0])
    rgbArray2=np.asarray(AdptTh(img2))
    img2 = Image.fromarray(rgbArray2)
    rgbArray[..., 1]=np.copy(rgbArray2)
    # image 2 saved as layer 1


    max_mrc=np.max(in3)
    min_mrc=np.min(in3)
    rgbArray2 = (254*((in3-min_mrc)/(max_mrc-min_mrc))).astype(np.uint8 )

    gzw=w//10
    gzh=h//10
    cl=(0.01*np.mean(rgbArray2))
    rgbArray2=apply_CLAHEMono(rgbArray2,gzw,gzh,cl)
    rgbArray1 = np.zeros((w,h,1), 'uint8')
    rgbArray1[..., 0]=np.copy(rgbArray2)
    #rgbArray1[..., 0]=cv2.flip((rgbArray1[..., 0]),0)
    img3 = Image.fromarray(rgbArray1[..., 0])
    rgbArray2=np.asarray(AdptTh(img3))
    rgbArray[..., 2]=np.copy(rgbArray2)
    img3 = Image.fromarray(rgbArray2)
    # image 3 saved as layer 3


    rgbArray1 = np.zeros((w,h,1), 'uint8')
    gzw=w//2
    gzh=h//2
    cl=(0.25*np.mean(rgbArray))
    #print(gzw,gzh,cl)
    rgbArray= apply_CLAHE(rgbArray,gzw=gzw,gzh=gzh,cl=cl)
    img = Image.fromarray(rgbArray)
    # Smooth the whole image
    img=adjust_sharpness(img,0.2)
    #img=Resize_imgAspect(img,owidth)
    dd="/".join(dest.split("/")[0:-2])+"/"
    img1.save(dd+"png0/"+outfl)
    img2.save(dd+"png1/"+outfl)
    img3.save(dd+"png2/"+outfl)
    img.save(dest+outfl)
    img.close()
    img1.close()
    img2.close()
    img3.close()

def get_shape(img):
    try:
        w,h,l=np.shape(np.asarray(img))
    except:
        w,h=np.shape(np.asarray(img))
    return (w,h)

# Takes an absolute file path and returns the name of the file without th extension
def filepath_to_name(full_name):
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    return file_name

def Resize_img(image,w2=0,h2=0):
    w,h=get_shape(image)
    if w2==0:
        w2=w
    if h2==0:
        h2=h
    imResize=image.resize((w2,h2), Image.NEAREST) #ANTIALIAS)
    return  (imResize)


def Resize_imgAspect(image,w2=0):
    mywidth=2048
    if w2==0:
        w2=mywidth
    j,m=get_shape(image)
    minj=np.max((j,m))
    wpercent=(w2/float(minj))
    if j==minj:
        hsize=int((float(image.size[0])*float(wpercent)))
        if hsize %2 != 0:
            hsize+=1
        img=image.resize((w2,hsize), Image.NEAREST)
    else:
        wsize=int((float(image.size[1])*float(wpercent)))
        if wsize %2 != 0:
            wsize+=1
        img=image.resize((wsize,w2), Image.NEAREST)

    return(img)

def load_image(path):
    try:
        image = cv2.cvtColor(cv2.imread(path,-1), cv2.COLOR_BGR2RGB)
    except:
        imy=cv2.imread(path,0)
        image=cv2.cvtColor(imy, cv2.COLOR_RGB2BGR)
    return image


def read_mrcfile(fln):
    import mrcfile
    print(fln)
    if(mrcfile.validate(fln)):
        mrc=mrcfile.open(fln).data
        max_mrc=np.max(mrc)
        min_mrc=np.min(mrc)
        #print(max_mrc,min_mrc)
        mrc2=(254*((mrc-min_mrc)/(max_mrc-min_mrc))).astype(np.uint8 )
        return np.asarray(mrc2)
    else:
        mrc=np.zeros((20,20), 'uint8')
        max_mrc=0
        min_mrc=0
        print("Invalid mrc file")
        return np.asarray(mrc)

def read_anymrcfile(fln):
    import mrcfile
    print(fln)
    mrc=mrcfile.open(fln,permissive=True).data
    max_mrc=np.max(mrc)
    min_mrc=np.min(mrc)
    #print(mrc.shape,max_mrc,min_mrc)
    mrc2=(254*((mrc-min_mrc)/(max_mrc-min_mrc))).astype(np.uint8 )
    return np.asarray(mrc2)


def one_hot_it(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes

    # Arguments
        label: The 2D array segmentation image label
        label_values

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    # st = time.time()
    # w = label.shape[0]
    # h = label.shape[1]
    # num_classes = len(class_dict)
    # x = np.zeros([w,h,num_classes])
    # unique_labels = sortedlist((class_dict.values()))
    # for i in range(0, w):
    #     for j in range(0, h):
    #         index = unique_labels.index(list(label[i][j][:]))
    #         x[i,j,index]=1
    # print("Time 1 = ", time.time() - st)

    # st = time.time()
    # https://stackoverflow.com/questions/46903885/map-rgb-semantic-maps-to-one-hot-encodings-and-vice-versa-in-tensorflow
    # https://stackoverflow.com/questions/14859458/how-to-check-if-all-values-in-the-columns-of-a-numpy-matrix-are-the-same
    semantic_map = []
    for colour in label_values:
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    # print("Time 2 = ", time.time() - st)

    return semantic_map

def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,1])

    # for i in range(0, w):
    #     for j in range(0, h):
    #         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
    #         x[i, j] = index

    x = np.argmax(image, axis = -1)
    return x


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """

    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,3])
    # colour_codes = label_values
    # for i in range(0, w):
    #     for j in range(0, h):
    #         x[i, j, :] = colour_codes[int(image[i, j])]

    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


def adjust_sharpness(input_image, factor):
    enhancer_object=ImageEnhance.Sharpness(input_image)
    out1=enhancer_object.enhance(factor)
    out2=ImageEnhance.Contrast(out1)
    out3=out2.enhance(1.5)
    out4=ImageEnhance.Brightness(out3)
    return out4.enhance(0.9)

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v=np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower=int(max(0, (1.0 - sigma) * v))
    upper=int(min(128, (1.0 + sigma) * v))
    edged=cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def cvtSLog(image=None):
    im=np.asarray(image)
    max_im=np.max(im)
    min_im=np.min(im)
    #print(max_im,min_im)
    imx=(254*((im-min_im)/(max_im-min_im))).astype(np.uint8 )
    img=np.log(1+imx)/np.log(2.0)  # use base 2 log
    max_im=np.max(img)
    min_im=np.min(img)
    #print(max_im,min_im)
    im=(254*((img-min_im)/(max_im-min_im))).astype(np.uint8 )
    return(im)

def save_imageSz(img,fl):
    #print(np.shape(img))
    w,h=np.shape(img)
    try:
       os.mkdir('./Cmetadata/')
    except:
       a=0
    import pickle    # Let us ensure that dimensions match that of the mrc file - first write
    p=(w,h)
    pickle.dump(p, open("./Cmetadata/"+fl+'_OrgSz.p', 'wb'))

def Watershed(img=None):
    import numpy as np
    import cv2
    from matplotlib import pyplot as plt

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3,3),np.uint8 )
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8 (sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    #save_mximg(unknown,filename="fig_test.png")
def checkincircle(center=(0,0),radius=50,pnt=(0,0)):
    if (np.sqrt((pnt[0] - center[0]) ** 2 + (pnt[1] - center[1]) ** 2) < radius):
        return True
    else:
        return False

def PrepareMRC(des="./png/",src="./mrc/"):
    import os,shutil,sys
    mrcpth=src
    try:
        os.mkdir( des)
    except:
        a=0



    def write_heatMap(img,outfl,dest,owidth):
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('jet')
        rgba_img = cmap(img)
        rgb_img = np.delete(rgba_img, 3, 2)
        #print(rgb_img.shape)
        img=Image.fromarray(rgb_img, 'RGB')
        img=Resize_imgAspect(img,owidth)
        img.save(dest+outfl)
    def nmzImg(img):
        mx_img=np.max(img)
        mn_img=np.min(img)
        return (254*((img-mn_img)/(mx_img-mn_img))).astype(np.uint8 )

    mrc=(os.listdir(mrcpth))
    cont=np.size(mrc)
    cnt=1

    try:
        os.mkdir( des)
    except:
        a=0
    print("Creating Prediction data from "+str(cont)+" mrc files in folder "+mrcpth+" to -> "+des)
    from os import path
    for fln in (mrc):
        if path.exists(des+'/'+fln.replace(".mrc",".png")):
            continue
        #fln='HCN1apo_0002_2xaligned.mrc'
        fltyp=os.path.splitext(fln)[1]
        fl=os.path.splitext(fln)[0]
        #fl=".".join(fl)
        if (fltyp==".jpg"):
            im=Image.open(des+"labels/"+fln)
            rgb_im=np.asarray(im.convert('RGB'))
            os.remove(des+"labels/"+fln)
            rgb_im[rgb_im>=170]=255
            rgb_im[rgb_im<170]=0

            fln=fl+'.png'
            im= Image.fromarray(rgb_im)
            im.save(des+"labels/"+fln)
            im.close()

        try:
            img=read_anymrcfile(mrcpth+'/'+fln.replace(".png",".mrc"))   # We are looking for mrc files in the folder
        except:
            print('Corrupt mrc, ignoring')
            continue
        print(fln)

        save_imageSz(np.asarray(img),fl)
        #img=Resize_img(Image.fromarray(img),w2=1024,h2=1024)   #save_mximg(img,filename="fig1.png")
        #img=np.uint8(img)
        img=np.copy(cv2.equalizeHist(img))
        #imf=ndimage.gaussian_filter(img,3)
        #img=np.uint8(imf)
        #img=np.copy(cv2.bilateralFilter(img,d=13,sigmaColor=5, sigmaSpace=5))
        d=np.max((np.shape(img)))
        d=d//640
        img=np.copy(cv2.bilateralFilter(img,d=d,sigmaColor=12, sigmaSpace=12))
        img=nmzImg(img)
        #clahe = cv2.createCLAHE(clipLimit=0, tileGridSize=(30,30))
        #img = clahe.apply(img)
        #img=np.copy(cv2.equalizeHist(img))
        #save_mximg(img,filename="fig3.png")
        if np.max(img)>0:
            # Contrast stretching
            imag=CutMedianTH(Image.fromarray(img))
            imag=nmzImg(imag)
            imgx=np.asarray(imag)
            imgx=nmzImg(imgx)#save_mximg(imgx,filename="fig4.png")
            cx=np.std(imgx)
            imgx=cv2.adaptiveThreshold(imgx,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,255,-0.1*cx)
            #output_width=512  # For training image is compressed to 512x512
            #output_height=np.shape(imgx)[0]
            write_img3(in1=np.asarray(imgx),in2=np.asarray(imag),in3=np.asarray(img),outfl=fl+".png",dest=des+"/")
            #write_heatMap(imgx,outfl=fl+".png",dest=dest+"/",owidth=output_width)

def get_label_info(csv_path):
    """
    Retrieve the class names and label values for the selected dataset.
    Must be in CSV format!

    # Arguments
        csv_path: The file path of the class dictionairy

    # Returns
        Two lists: one for the class names and the other for the label values
    """
    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")

    class_names = []
    label_values = []
    with open(csv_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
        # print(class_dict)
    return class_names, label_values

def DrawPoly(image=None):
    PolyA_array=[]
    PolyR_array=[]
    import cv2
    img = cv2.imread(image)
   
    from time import time
    boxes = []
    print(' --------------------------------------')
    print('|        Press "q" key to quit         |')
    print('|     "a" key to append to selection   |')
    print('|    "r" key to remove from selection  |')
    print('|     +/- to zoom in and out image     |')
    print(' --------------------------------------')
    mulf=1
    def on_mouse(event, x, y, flags, params):
 
        if event == cv2.EVENT_LBUTTONDOWN:
            #print('Start Mouse Position: '+str(x)+', '+str(y))
            sbox = [x, y]
            boxes.append(sbox)
            # print count
            # print sbox

        elif event == cv2.EVENT_LBUTTONUP:
            #print('End Mouse Position: '+str(x)+', '+str(y))
            ebox = [x, y]
            boxes.append(ebox)
            #print(boxes)
            crop = img[boxes[-2][1]:boxes[-1][1],boxes[-2][0]:boxes[-1][0]]
            if ((crop.shape[0]!=0) and (crop.shape[1]!=0)):
                cv2.imshow('crop',crop)
                k =  cv2.waitKey(0)
                if k== ord('a'):
                    PolyA_array.append([boxes[-2][1],boxes[-1][1],boxes[-2][0],boxes[-1][0]])
                if k== ord('r'):
                    PolyR_array.append([boxes[-2][1],boxes[-1][1],boxes[-2][0],boxes[-1][0]])
                cv2.destroyWindow('crop')
 
            
    while(1):
 
        cv2.namedWindow('real image')
        cv2.setMouseCallback('real image', on_mouse, 0)
        cv2.imshow('real image', img)
        kk=cv2.waitKey(0)
        if  kk== ord('q'):
                cv2.destroyAllWindows()
                break
        if kk == ord('+'):
            img = cv2.resize(img, None, fx = 4,fy = 4)
            mulf=mulf/4
            cv2.destroyAllWindows()
        if kk == ord('-'):
            img = cv2.resize(img, None, fx = 0.25,fy = 0.25)
            mulf=mulf*4
            cv2.destroyAllWindows()
    if PolyA_array==[]:
        PolyA_array=[[0,img.shape[1],0,img.shape[0]]]
    return(mulf,PolyA_array,PolyR_array)

def LabelMRC(image=None,mrcsrc="./mrc/",labelpath='',TrData='TrData.txt',debug=False):
     
 
    print("-------------------------------------------------------------------------")
    print('CASSPER Label Toolkit')
    print('USAGE:')
    print('Use the following keyboard commands to navigate. Click mouse Left Button')
    print('to select regions for labels.')
    print('-------------------------------------------------------------------------')
    print()
    print('"u" - move up    in the window')
    print('"d" - move down  in the window')
    print('"l" - move left  in the window')
    print('"r" - move right in the window')
    print('"c" - toggle label colour/type')
    print('"s" - skip last  click of mouse')
    print('"g" - get last   skipped click')
    print('"e" - erase last selection by click')
    print('"v" - Full Image view')
    print('"+" - increase window size')
    print('"-" - decrease window size')
    print('"q" - quit this image and move to the next')
    print('-------------------------------------------------------------------------')
    input("Press Enter to continue...")
    cutwidth=256
    skipval=cutwidth//4
    checksetup()
    class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
    class_names_list, label_values=get_label_info("models/class_dict.csv")
    num_classes=len(label_values)
    ColourCodes=[]
    for i in range(num_classes):
        ColourCodes.append([class_names_list[i],np.array(label_values[i])])
        PCode=0
    if debug==True:
        print(ColourCodes)
    metafold=mrcsrc+"/meta/"
    if labelpath=='':
        labelpath=mrcsrc+"/labels/"
    try:
        os.mkdir(metafold)
    except:
        a=0
    try:
        pt=labelpath.split('/')
        pi=''
        for i in pt:
            if (pi==''):
                pi="%s/"%i
            else:
                pi="%s%s/"%(pi,i)
            if (pi !='./'):
                try:
                    os.mkdir(pi)
                except:
                    a=0
    except:
        print("can't create %s for %s"%(pi,pt))
        return(0)
        a=0

    global final,img,stack,view_on
    global pix_val, clicks
    #global minarea,maxarea
    view_on=False
    def click_and_crop(event, x, y, flags, param):
        global img,imx,final
        global pix_val, clicks
        global maxarea,minarea,window_range
        window_range = cv2.getTrackbarPos('maskwindow',fln)
        minarea = cv2.getTrackbarPos('minarea',fln)
        maxarea = cv2.getTrackbarPos('maxarea',fln)
        img=imx[oldx-cutwidth:oldx+cutwidth,oldy-cutwidth:oldy+cutwidth]
        if event == cv2.EVENT_LBUTTONDOWN:
            try:
                pix_val = img[y,x]
                if final !=[]:
                        fflag=False
                        for i in range(len(final)):
                            if (np.sum(np.abs(final[i][0] -pix_val)) <= window_range):
                                fflag=True
                                if  PCode !=final[1][-1]:
                                    final[i]=[pix_val,window_range,minarea,maxarea,PCode]
                                    #print([pix_val,window_range,minarea,maxarea,PCode])


                            else:
                                if i==len(final)-1 and fflag==False:
                                    final.append([pix_val,window_range,minarea,maxarea,PCode])

                else:
                    final.append([pix_val,window_range,minarea,maxarea,PCode])
                clicks.append((x,y))
                show_canvas(img=img,fln=fln)
            except:
                print('Please exit view mode and try ...')
                pass

    def show_canvas(img=None,save=None,fln=None):
        cv2.namedWindow(fln,cv2.WINDOW_NORMAL)
        global maxarea,minarea,window_range
        window_range = cv2.getTrackbarPos('maskwindow',fln)
        minarea = cv2.getTrackbarPos('minarea',fln)
        maxarea = cv2.getTrackbarPos('maxarea',fln)
        global BGClr,view_on
        imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        canvas = np.zeros(img.shape, np.uint8)
        BGClr=[np.flip(c[1])  for c in ColourCodes if c[0].upper()=="Background".upper()]
        canvas[:] = BGClr #[255,0,0]
        im = np.copy(img)
        for [pix_val,window_range,minarea,maxarea,PCode] in final:
            lower_blue = pix_val - window_range
            upper_blue = pix_val + window_range
            mask_blue = cv2.inRange(im, lower_blue, upper_blue)
            try:
                contours, h = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            except:
                a, contours, h = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


            for c in contours:
                #print(cv2.contourArea(c))
                #print(minarea,maxarea)

                if cv2.contourArea(c)>=minarea and cv2.contourArea(c)<maxarea:
                    PCCode=ColourCodes[PCode][1]
                    cv2.drawContours(canvas,[c],0,(int(PCCode[2]),int(PCCode[1]),int(PCCode[0])),-1)
                    if ((str(np.flip(PCCode)) != str(BGClr[0])) and debug==True):
                        cv2.drawContours(im, [c], 0, (255, 255, 255), -1)

        if save==None:
            im_disp = cv2.hconcat([im, canvas])
            if view_on==True:
                view_on=False
                cv2.namedWindow("%s"%fln,cv2.WINDOW_NORMAL)
                cv2.imshow("%s"%fln,im_disp)
                cv2.waitKey()
            else:
                cv2.imshow(fln,im_disp)
        else:
            cv2.imwrite("%s/%s"%(save,fln),canvas)
            if debug==True:
                cv2.imwrite("%s/O_%s"%(save,fln),im)
            save=None

    dest="/tmp/P_files"
    try:
        os.mkdir(dest)
    except:
        a=0
    try:
        os.mkdir('/tmp/train')
    except:
        a=0
    copytree(mrcsrc, "/tmp/train/",Tfiles=getfiles(TrData))
    PrepareMRC(des= dest,src="/tmp/train/")
    shutil.rmtree('/tmp/train/')

    if (image==None):
        files=(os.listdir(dest))
        cont=np.size(files)
        print("Preparing for testing on "+str(np.int(cont))+"  files...")
        imcnt=0
        final=[]
        stack=[]
        Tfl=sgetfiles(TrData)
        for fln in (files):
            if fln in Tfl:

                global img,imx
                imcnt+=1
                pix_val=0
                clicks=[]
                try:
                    with open(metafold+'f_%s'%fln.replace('.png','.dat'), 'rb') as filehandle:
                        final=pickle.load(filehandle)
                    with open(metafold+'s_%s'%fln.replace('.png','.dat'), 'rb') as filehandle:
                        stack=pickle.load(filehandle)
                except:
                    a=0
                print("Testing image " + fln)
                imy=load_image(dest+'/'+fln)
                cv2.imwrite("/tmp/%s"%fln,imy)
                #imy=load_image('/tmp/%s'%fln)
                match_star_if_found(file='/tmp/%s'%(fln),sfdes=mrcsrc+"/sf",Box=False)
                mulf,Select_r,Reject_r=DrawPoly(image='/tmp/%s'%fln)
                imx=load_image('/tmp/%s'%fln)
                im=np.zeros_like(imx)
                for m in Select_r:
                    m=mulf*np.array(m)
                    i,j,k,l = m
                    print(i,j,k,l)
                    im[i:j,k:l]=imx[i:j,k:l]
                for m in Reject_r:
                    m=mulf*np.array(m)
                    i,j,k,l = m
                    im[i:j,k:l]=0
                imx=im.copy()
                os.remove('/tmp/%s'%fln)
                #imx=cv2.cvtColor(imxx,cv2.COLOR_BGR2GRAY)
                if debug==True:
                    print(imx.shape)
                sx,sy,cl=imx.shape
                if min(sx,sy)<2*cutwidth:
                    cutwidth=min(sx,sy)//4
                img=imx[sx//2-cutwidth:sx//2+cutwidth,sy//2-cutwidth:sy//2+cutwidth]

                oldx=sx//2
                oldy=sy//2
                show_canvas(img=img,fln=fln)
                cv2.createTrackbar('maskwindow',fln,30,900,callback)
                cv2.createTrackbar('minarea',fln,140,1000,callback)
                cv2.createTrackbar('maxarea',fln,1640,6000,callback)
                while(1):
                    cv2.setMouseCallback(fln, click_and_crop)

                    k=None
                    k = cv2.waitKey() & 0xFF
                    if k == ord('q'):
                        print(f"{bcolors.WARNING}Finalising your selections to entire mrc. This may take some time depending on the size of the mrc file. Please hold on...{bcolors.ENDC}")
                        show_canvas(img=imx,fln=fln,save=labelpath)
                        break
                    if k == ord('e'):      # Erase these
                        if (final !=[]):
                            img=imx[oldx-cutwidth:oldx+cutwidth,oldy-cutwidth:oldy+cutwidth]
                            final=final[:-1]
                            while stack !=[]:
                                [final.append([a,b,c,d,e]) for [a,b,c,d,e] in stack[-1:]]
                                stack=stack[:-1]
                            show_canvas(img=img,fln=fln)

                        else:
                            continue
                    if k == ord('s'):      # Skip these
                        if (final !=[]):
                            img=imx[oldx-cutwidth:oldx+cutwidth,oldy-cutwidth:oldy+cutwidth]
                            [stack.append([a,b,c,d,e]) for [a,b,c,d,e] in final[-1:]]
                            final=final[:-1]
                            show_canvas(img=img,fln=fln)

                        else:
                            continue
                    if k == ord('c'):      # Change label code these
                            PCode=(PCode+1)%len(ColourCodes)
                            img=imx[oldx-cutwidth:oldx+cutwidth,oldy-cutwidth:oldy+cutwidth]
                            show_canvas(img=img,fln=fln)
                            print("Particles marked as %s"%ColourCodes[PCode][0])

                    if k == ord('g'):      # Get these
                        if (stack !=[]):
                            [final.append([a,b,c,d,e]) for [a,b,c,d,e] in stack[-1:]]
                            stack=stack[:-1]
                            img=imx[oldx-cutwidth:oldx+cutwidth,oldy-cutwidth:oldy+cutwidth]
                            show_canvas(img=img,fln=fln)

                        else:
                            continue
                    if k == ord('+'):      # Skip these
                        if (cutwidth+256 <= min(sx//2,sy//2)):
                            cutwidth=cutwidth+256
                        else:
                            cutwidth=min(sx//2,sy//2)
                        skipval=cutwidth//4
                        oldx=sx//2
                        oldy=sy//2
                        img=imx[oldx-cutwidth:oldx+cutwidth,oldy-cutwidth:oldy+cutwidth]
                        show_canvas(img=img,fln=fln)

                    if k == ord('-'):      # Skip these
                        if (cutwidth-256 > 0):
                            cutwidth=cutwidth-256
                        else:
                            cutwidth=256
                        skipval=cutwidth//4
                        oldx=sx//2
                        oldy=sy//2

                        img=imx[oldx-cutwidth:oldx+cutwidth,oldy-cutwidth:oldy+cutwidth]
                        show_canvas(img=img,fln=fln)
                    if k == ord('v'):      # Show full image
                        print(f"{bcolors.WARNING}You have selected to view the entire mrc. This may take some time depending on the size of the mrc file. Please hold on...{bcolors.ENDC}")
                        if view_on == True:
                            view_on=False
                            img=imx[oldx-cutwidth:oldx+cutwidth,oldy-cutwidth:oldy+cutwidth]
                            show_canvas(img=img,fln=fln)
                        else:
                            view_on=True
                            show_canvas(img=imx,fln=fln)

                    if k== ord('l'):
                        if oldy>cutwidth+skipval:
                            oldy=oldy-skipval
                            img=imx[oldx-cutwidth:oldx+cutwidth,oldy-cutwidth:oldy+cutwidth]
                            show_canvas(img=img,fln=fln)
                        else:
                            oldy=oldy+cutwidth+skipval
                            continue
                    if k== ord('r'):
                        if oldy<=sy-cutwidth-skipval:
                            oldy=oldy+skipval
                            img=imx[oldx-cutwidth:oldx+cutwidth,oldy-cutwidth:oldy+cutwidth]
                            show_canvas(img=img,fln=fln)
                        else:
                            oldy=sy-cutwidth-skipval
                            continue
                    if k== ord('d'):
                        if oldx<=sx-cutwidth-skipval:
                            oldx=oldx+skipval
                            img=imx[oldx-cutwidth:oldx+cutwidth,oldy-cutwidth:oldy+cutwidth]
                            show_canvas(img=img,fln=fln)
                        else:
                            oldx=sx-cutwidth-skipval
                            continue
                    if k== ord('u'):
                        if oldx>=cutwidth+skipval:
                            oldx=oldx-skipval
                            img=imx[oldx-cutwidth:oldx+cutwidth,oldy-cutwidth:oldy+cutwidth]
                            show_canvas(img=img,fln=fln)
                        else:
                            oldx=cutwidth+skipval
                            oldx=cutwidth+skipval
                            continue

                if final==[]:
                    continue
                else:
                    with open(metafold+'f_%s'%fln.replace('.png','.dat'), 'wb') as filehandle:
                        pickle.dump(final, filehandle)
                    with open(metafold+'s_%s'%fln.replace('.png','.dat'), 'wb') as filehandle:
                        pickle.dump(stack, filehandle)
                    cv2.destroyWindow(fln)
                    imyy=cv2.cvtColor(imy, cv2.COLOR_RGB2BGR)

    cv2.destroyAllWindows()

def PredictMRC(image=None,use_model='default',model_path='models/',model_name='BestFr_InceptionV4_model_FRRN-B.ckpt',crop_height=512,crop_width=512,model="FRRN-B",dataset="./png/",mrcsrc="./mrc/",Savestar=True,CrossMatch=False,Box=True):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    import gc
    gc.collect()

    checksetup()
    print("------------------------------------------------------------------------------------------------------")
    print("This is a complete rewrite of CASSPER code for testing new mrc files")
    #print("Author: Ninan Sajeeth Philip - ninansajeethphilip@airis4d.com")
    print("------------------------------------------------------------------------------------------------------")
    print("Command is PredictMRC(image=None,use_model='default',model_path='/models/',model_name='BestFr_InceptionV4_model_FRRN-B.ckpt',crop_height=512,crop_width=512,model='FRRN-B',dataset='./png/',mrcsrc='./mrc/',Savestar=True,CrossMatch=False,Box=True)")
    print("------------------------------------------------------------------------------------------------------")
    print("")
    global class_names_list, label_values
    class_names_list, label_values=get_label_info("models/class_dict.csv")

    num_classes=len(label_values)

    print("\n***** Begin prediction *****")

    # Initializing network
    config=tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True
    sess=tf.compat.v1.Session(config=config)
    tf.compat.v1.disable_eager_execution()
    net_input=tf.compat.v1.placeholder(tf.float32,shape=[None,None,None,3])
    net_output=tf.compat.v1.placeholder(tf.float32,shape=[None,None,None,num_classes])

    network, _=build_model(model, net_input=net_input,
                                        num_classes=num_classes,
                                        crop_width=crop_width,
                                        crop_height=crop_height,
                                        is_training=False)


    print('Loading model weights')
    saver=tf.compat.v1.train.Saver(max_to_keep=1000)
    sess.run(tf.compat.v1.global_variables_initializer())
    if use_model=='default':
        model_name="CrossPro.ckpt"
        print('New model from your training is used for predictions')
    else:
        print('Predictions are made using the model: %s'%model_name)

    
    mymodel="%s%s"%(model_path,model_name)
    saver.restore(sess,mymodel)
    print('loaded model..')
    des=dataset
    try:
        os.mkdir( des)
    except:
        a=0
    try:
        os.mkdir( des+"P_files/" )
    except:
        print("Folder P_files/ already exist. Files will be appended to the folder")
    try:
        os.mkdir(des+'Predict_labels/')
    except:
        a=0
    dest=des+"P_files"
    PrepareMRC(src=mrcsrc,des=dest)
    st=time.time()

    if (image==None):
        files=(os.listdir(dest))
        cont=np.size(files)
        print("Preparing for testing on "+str(np.int(cont))+"  files...")
        imcnt=0
        for fln in (files):
            imcnt+=1

            print("Testing image " + fln)

            mulf,Select_r,Reject_r=DrawPoly(image=dest+'/'+fln)
            imx=load_image(dest+'/'+fln)
            im=np.zeros_like(imx)
            for m in Select_r:
                m=mulf*np.array(m)
                i,j,k,l = m
                im[i:j,k:l]=imx[i:j,k:l]
            for m in Reject_r:
                m=mulf*np.array(m)
                i,j,k,l = m
                im[i:j,k:l]=0
            loaded_image=im.copy()

            resized_image=cv2.resize(loaded_image, (crop_width, crop_width))
            input_image=np.expand_dims(np.float32(resized_image[:crop_height, :crop_width]),axis=0)/  255.0
            #input_image=np.expand_dims(np.float32(load_image(val_input_names[ind])[:crop_height, :crop_width]),axis=0)/255.0

            output_image=sess.run(network,feed_dict={net_input:input_image})


            output_image=np.array(output_image[0,:,:,:])
            output_image=reverse_one_hot(output_image)
            out_vis_image=colour_code_segmentation(output_image, label_values)
            file_name=os.path.splitext(fln)[0]
            #file_name=".".join(fl)
            cv2.imwrite("/tmp/%s_pred.png"%(file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
            scale_write_img(filename=os.path.splitext(fln)[0],pth=des,des="/Predict_labels/",src="/tmp/%s_pred.png"%(file_name))
            os.remove("/tmp/%s_pred.png"%(file_name))
            if Savestar==True:
                try:
                    os.mkdir(des+'StarFiles/')
                except:
                    a=0

                if imcnt <2:

                    global rd
                    global ed
                    global da
                    global kv
                    rd,ed,da,kv,wt,th=get_radius_erode(image=des+'Predict_labels/'+'%s.png'%(file_name),shimg=dest+'/'+fln)
                getstar(radius=rd,erode=ed,dial=da,kl=kv,wt=wt,th=th,image=des+'Predict_labels/'+'%s.png'%(file_name),simage=dest+'/'+fln,des=des+'StarFiles/',mrcsrc=mrcsrc,fln=fln,CrossMatch=CrossMatch,Box=Box)

    else:
        print("Testing image " + image)

        loaded_image=load_image(image)
        resized_image=cv2.resize(loaded_image, (crop_width, crop_width))
        input_image=np.expand_dims(np.float32(resized_image[:crop_height, :crop_width]),axis=0)/255.0

        output_image=sess.run(network,feed_dict={net_input:input_image})


        output_image=np.array(output_image[0,:,:,:])
        output_image=reverse_one_hot(output_image)
        out_vis_image=colour_code_segmentation(output_image, label_values)
        file_name=filepath_to_name(image)
        cv2.imwrite("%s_pred.png"%(file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
        scale_write_img(filename="%s_fpred.png"%(file_name),src="%s_pred.png"%(file_name))
        #im_dummy=circle_mark(file_name)

        if Savestar==True:
            try:
                os.mkdir(des+'StarFiles/')
            except:
                a=0
            rd,ed,da,kv,wt,th=get_radius_erode(image=des+'Predict_labels/'+'%s.png'%(file_name),shimg=image)
            getstar(radius=rd,erode=ed,dial=da,kl=kv,wt=wt,th=th,image=des+'Predict_labels/'+'%s.png'%(file_name),simage=dest+'/'+fln,des=des+'StarFiles/',mrcsrc=mrcsrc,fln=des+'Predict_labels/'+image,CrossMatch=CrossMatch,Box=Box)

    run_time=time.time()-st

    print("")
    print("Finished in %s seconds" %run_time)
    print("--------Bye-----------")



#--------------------------------Training Cycle ------------------------------------------
def evaluate_segmentation(pred, label, num_classes, score_averaging="weighted"):
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)

    prec = precision_score(flat_pred, flat_label, average=score_averaging)
    rec = recall_score(flat_pred, flat_label, average=score_averaging)
    f1 = f1_score(flat_pred, flat_label, average=score_averaging)

    iou = compute_mean_iou(flat_pred, flat_label)

    return global_accuracy, class_accuracies, prec, rec, f1, iou

# Print with time. To console or file
def LOG(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def getfiles(file='TrData.txt'):
    nlist=[]
    with open(file) as fp:
        namelist = fp.read().splitlines()
    for i in namelist:
        nlist.append(i)
    return(nlist)

def sgetfiles(file='TrData.txt',rep='.png'):
    nlist=[]
    with open(file) as fp:
        namelist = fp.read().splitlines()
    for i in namelist:
        pre, ext = os.path.splitext(i)
        newn=(pre + rep)
        nlist.append(newn)
    return(nlist)

def copytree(src, dst, Tfiles=[],symlinks=False, ignore=None):
    for item in Tfiles:    #os.listdir(src):
        s = os.path.join(src, item.split('/')[-1])
        d = os.path.join(dst, item.split('/')[-1])
        if os.path.isdir(s):
            try:
                shutil.copytree(s, d, symlinks, ignore)
            except:
                print("File %s Exist"%d)
        else:
            shutil.copy2(s, d)

import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    

    response = session.get(URL, params = { 'id' : id }, stream = True)
    if (response.status_code == 200):
        print("The request was a success!")
        token = get_confirm_token(response)
        print(token)
        if token:
           
            params = { 'id' : id, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)
        

        save_response_content(response, destination)
    elif (response.status_code == 404):
        print("Result not found!")
        # Code here will react to failed requests

def get_confirm_token(response):
    hrr=response.headers
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    for key, value in hrr.items():
        if key.startswith('Content-Type'):
            return True
 
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def prepare_data(dataset_dir='.png/',Tfolder='./mrc/'):
    train_input_names=[]
    train_output_names=[]
    val_input_names=[]
    val_output_names=[]
    test_input_names=[]
    test_output_names=[]
    try:
        shutil.rmtree(dataset_dir)
        os.mkdir(dataset_dir)
    except:
        a=0
    try:
            os.mkdir(dataset_dir + "/train/")
    except:
            a=0
    try:
            os.mkdir(dataset_dir + "/train_labels/")
    except:
            a=0
    try:
            os.mkdir(dataset_dir + "/val_labels/")
    except:
            a=0
    try:
            os.mkdir(dataset_dir + "/test_labels/")
    except:
            a=0
    try:
        os.mkdir('/tmp/png/')
        os.mkdir('/tmp/png/train')
        os.mkdir('/tmp/png/test')
    except:
        shutil.rmtree('/tmp/png/')
        os.mkdir('/tmp/png/')
        os.mkdir('/tmp/png/train')
        os.mkdir('/tmp/png/test')
    copytree(Tfolder, "/tmp/png/train/",Tfiles=getfiles('TrData.txt'))
    PrepareMRC(des= dataset_dir+"/train/",src="/tmp/png/train/")
    shutil.copytree(dst= dataset_dir+"/val/",src=dataset_dir+"/train/")
    try:
        copytree(Tfolder, 'tmp/png/test/',Tfiles=getfiles('TsData.txt'))
        copytree(Tfolder+'/labels/', dataset_dir+"/test_labels/",Tfiles=sgetfiles('TsData.txt','.png'))
        PrepareMRC(des= dataset_dir+"/test/",src="/tmp/png/test/")
    except:
        shutil.copytree(dst= dataset_dir+"/test/",src=dataset_dir+"/train/")
        copytree(Tfolder+'/labels/', dataset_dir+"/test_labels/",Tfiles=sgetfiles('TrData.txt','.png'))
    copytree(Tfolder+'/labels/', dataset_dir+"/train_labels/",Tfiles=sgetfiles('TrData.txt','.png'))
    copytree(Tfolder+'/labels/', dataset_dir+"/val_labels/",Tfiles=sgetfiles('TrData.txt','.png'))
    for file in os.listdir(dataset_dir + "/train"):
        cwd = os.getcwd()
        train_input_names.append(cwd + "/" + dataset_dir + "/train/" + file)
    for file in os.listdir(dataset_dir + "/train_labels"):
        cwd = os.getcwd()
        train_output_names.append(cwd + "/" + dataset_dir + "/train_labels/" + file)
    for file in os.listdir(dataset_dir + "/val"):
        cwd = os.getcwd()
        val_input_names.append(cwd + "/" + dataset_dir + "/val/" + file)
    for file in os.listdir(dataset_dir + "/val_labels"):
        cwd = os.getcwd()
        val_output_names.append(cwd + "/" + dataset_dir + "/val_labels/" + file)
    for file in os.listdir(dataset_dir + "/test"):
        cwd = os.getcwd()
        test_input_names.append(cwd + "/" + dataset_dir + "/test/" + file)
    for file in os.listdir(dataset_dir + "/test_labels"):
        cwd = os.getcwd()
        test_output_names.append(cwd + "/" + dataset_dir + "/test_labels/" + file)
    train_input_names.sort(),train_output_names.sort(), val_input_names.sort(), val_output_names.sort(), test_input_names.sort(), test_output_names.sort()
    shutil.rmtree('/tmp/png/')
    return train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names




# Count total number of parameters in the model
def count_params():
    total_parameters = 0
    '''for variable in tf.compat.v1.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("This model has %d trainable parameters"% (total_parameters))
    '''
# Randomly crop the image to a specific size. For data augmentation
def random_crop(image, label, crop_height, crop_width):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')

    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        x = random.randint(0, image.shape[1]-crop_width)
        y = random.randint(0, image.shape[0]-crop_height)

        if len(label.shape) == 3:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width, :]
        else:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width]
    else:
        raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (crop_height, crop_width, image.shape[0], image.shape[1]))

# Compute the average segmentation accuracy across all classes
def compute_global_accuracy(pred, label):
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)

# Compute the class-specific segmentation accuracy
def compute_class_accuracies(pred, label, num_classes):
    total = []
    for val in range(num_classes):
        total.append((label == val).sum())

    count = [0.0] * num_classes
    for i in range(len(label)):
        if pred[i] == label[i]:
            count[int(pred[i])] = count[int(pred[i])] + 1.0

    # If there are no pixels from a certain class in the GT,
    # it returns NAN because of divide by zero
    # Replace the nans with a 1.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / total[i])

    return accuracies


def compute_mean_iou(pred, label):

    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels);

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))


    mean_iou = np.mean(I / U)
    return mean_iou


def evaluate_segmentation(pred, label, num_classes, score_averaging="weighted"):
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)

    prec = precision_score(flat_pred, flat_label, average=score_averaging)
    rec = recall_score(flat_pred, flat_label, average=score_averaging)
    f1 = f1_score(flat_pred, flat_label, average=score_averaging)

    iou = compute_mean_iou(flat_pred, flat_label)

    return global_accuracy, class_accuracies, prec, rec, f1, iou


def compute_class_weights(labels_dir, label_values):
    '''
    Arguments:
        labels_dir(list): Directory where the image segmentation labels are
        num_classes(int): the number of classes of pixels in all images

    Returns:
        class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.

    '''
    image_files = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir) if file.endswith('.png')]

    num_classes = len(label_values)

    class_pixels = np.zeros(num_classes)

    total_pixels = 0.0

    for n in range(len(image_files)):
        image = imread(image_files[n])

        for index, colour in enumerate(label_values):
            class_map = np.all(np.equal(image, colour), axis = -1)
            class_map = class_map.astype(np.float32)
            class_pixels[index] += np.sum(class_map)


        print("\rProcessing image: " + str(n) + " / " + str(len(image_files)))
        sys.stdout.flush()

    total_pixels = float(np.sum(class_pixels))
    index_to_delete = np.argwhere(class_pixels==0.0)
    class_pixels = np.delete(class_pixels, index_to_delete)

    class_weights = total_pixels / class_pixels
    class_weights = class_weights / np.sum(class_weights)

    return class_weights

# Compute the memory usage, for debugging
def memory():
    import os
    import psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # Memory use in GB
    print('Memory usage in GBs:', memoryUse)



def TrainMRC(num_epochs=300,epoch_start_i=0,checkpoint_step=5,validation_step=1,image=None,continue_training=False,dataset='./png',model_path='models/',TrDFolder='./mrc',batch_size=1,num_val_images=20,h_flip=False,v_flip=False,brightness=None,rotation=None):
    model='FRRN-B'
    frontend='InceptionV4'
    crop_height=512
    crop_width=512
    checksetup()

    try:
        shutil.rmtree(dataset)
    except:
        a=0
    try:
        os.mkdir('./models/checkpoints')
    except:
        print('checkpoints folder exists')
    os.mkdir(dataset)
    def data_augmentation(input_image, output_image):
        # Data augmentation
        input_image, output_image=random_crop(input_image, output_image, crop_height, crop_width)

        if h_flip and random.randint(0,1):
            input_image=cv2.flip(input_image, 1)
            output_image=cv2.flip(output_image, 1)
        if v_flip and random.randint(0,1):
            input_image=cv2.flip(input_image, 0)
            output_image=cv2.flip(output_image, 0)
        if brightness:
            factor=1.0 + random.uniform(-1.0*brightness, brightness)
            table=np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
            input_image=cv2.LUT(input_image, table)
        if rotation:
            angle=random.uniform(-1*rotation, rotation)
        if rotation:
            M=cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
            input_image=cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
            output_image=cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)

        return input_image, output_image

    # Get the names of the classes so we can record the evaluation results
    class_names_list, label_values=get_label_info("models/class_dict.csv")
    class_names_string=""
    for class_name in class_names_list:
        if not class_name==class_names_list[-1]:
            class_names_string=class_names_string + class_name + ", "
        else:
            class_names_string=class_names_string + class_name

    num_classes=len(label_values)

    config=tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True
    sess=tf.compat.v1.Session(config=config)


    # Compute your softmax cross entropy loss
    net_input=tf.compat.v1.placeholder(tf.float32,shape=[None,None,None,3])
    net_output=tf.compat.v1.placeholder(tf.float32,shape=[None,None,None,num_classes])

    network, init_fn=build_model(model_name=model, frontend=frontend, net_input=net_input,    num_classes=num_classes, crop_width=crop_width, crop_height=crop_height, is_training=True)

    loss=tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=tf.stop_gradient(net_output)))

    opt=tf.compat.v1.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(loss, var_list=[var for var in tf.compat.v1.trainable_variables()])

    saver=tf.compat.v1.train.Saver(max_to_keep=1000)
    sess.run(tf.compat.v1.global_variables_initializer())

    count_params()

    # If a pre-trained ResNet is required, load the weights.
    # This must be done AFTER the variables are initialized with sess.run(tf.global_variables_initializer())
    if init_fn is not None:
        init_fn(sess)


    avg_loss_per_epoch=[]
    avg_scores_per_epoch=[]
    avg_iou_per_epoch=[]
    avg_f1_per_epoch=[]
    # Which validation images do we want
    val_indices=[]

    # Load a previous checkpoint if desired
    model_checkpoint_name=model_path+"checkpoints/latest_model_"+ frontend+'_'+ model + "_" +".ckpt"
    old_epoch=0
    if continue_training:
        print('Loaded latest model checkpoint')
        saver.restore(sess, model_checkpoint_name)
        old_epoch=epoch_start_i

    # Load the data
    print("Loading the data ...")
    train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names=prepare_data(dataset_dir=dataset)



    print("\n***** Begin training *****")
    print("Dataset -->", dataset)
    print("Model -->", model)
    print("Crop Height -->", crop_height)
    print("Crop Width -->", crop_width)
    print("Num Epochs -->", num_epochs)
    print("Batch Size -->", batch_size)
    print("Num Classes -->", num_classes)

    print("Data Augmentation:")
    print("\tVertical Flip -->", v_flip)
    print("\tHorizontal Flip -->", h_flip)
    print("\tBrightness Alteration -->", brightness)
    print("\tRotation -->", rotation)
    print("")
    num_vals=min(num_val_images, len(val_input_names))

    # Set random seed to make sure models are validated on the same validation images.
    # So you can compare the results of different models more intuitively.
    random.seed(16)
    val_indices=random.sample(range(0,len(val_input_names)),num_vals)

    # Do the training here
    for epoch in range(epoch_start_i, num_epochs):

        current_losses=[]

        cnt=0

        # Equivalent to shuffling
        id_list=np.random.permutation(len(train_input_names))

        num_iters=int(np.floor(len(id_list) / batch_size))
        st=time.time()
        epoch_st=time.time()
        for i in range(num_iters):
            st=time.time()

            input_image_batch=[]
            output_image_batch=[]

            # Collect a batch of images
            for j in range(batch_size):
                index=i*batch_size + j
                id=id_list[index]
                input_image=load_image(train_input_names[id])
                output_image=load_image(train_output_names[id])
                with tf.device('/cpu:0'):
                    input_image, output_image=data_augmentation(input_image, output_image)


                    # Prep the data. Make sure the labels are in one-hot format
                    input_image=np.float32(input_image) / 255.0
                    output_image=np.float32(one_hot_it(label=output_image, label_values=label_values))

                    input_image_batch.append(np.expand_dims(input_image, axis=0))
                    output_image_batch.append(np.expand_dims(output_image, axis=0))

            if batch_size==1:
                input_image_batch=input_image_batch[0]
                output_image_batch=output_image_batch[0]
            else:
                input_image_batch=np.squeeze(np.stack(input_image_batch, axis=1))
                output_image_batch=np.squeeze(np.stack(output_image_batch, axis=1))

            # Do the training
            _,current=sess.run([opt,loss],feed_dict={net_input:input_image_batch,net_output:output_image_batch})
            current_losses.append(current)
            cnt=cnt + batch_size
            if cnt % 20==0:
                string_print="Epoch=%d Count=%d Current_Loss=%.4f Time=%.2f"%(epoch,cnt,current,time.time()-st)
                LOG(string_print)
                st=time.time()

        mean_loss=np.mean(current_losses)
        avg_loss_per_epoch.append(mean_loss)

        # Create directories if needed
        #if not os.path.isdir("%s%s/%04d"%(model_path,"checkpoints",epoch)):
        #    os.makedirs("%s%s/%04d"%(model_path,"checkpoints",epoch))

        # Save latest checkpoint to same file name
        print("Saving latest checkpoint")
        saver.save(sess,model_checkpoint_name)

        #if val_indices !=0 and epoch % checkpoint_step==0:
        #    print("Saving checkpoint for this epoch")
        #    saver.save(sess,"%s%s/%04d/model.ckpt"%(model_path,"checkpoints",epoch))


        if epoch % validation_step==0:
            print("Performing validation")
        #    target=open("%s%s/%04d/val_scores.csv"%(model_path,"checkpoints",epoch),'w')
        #    target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))


            scores_list=[]
            class_scores_list=[]
            precision_list=[]
            recall_list=[]
            f1_list=[]
            iou_list=[]


            # Do the validation on a small set of validation images
            for ind in val_indices:

                input_image=np.expand_dims(np.float32(load_image(val_input_names[ind])[:crop_height, :crop_width]),axis=0)/255.0
                gt=load_image(val_output_names[ind])[:crop_height, :crop_width]
                gt=reverse_one_hot(one_hot_it(gt, label_values))

                st=time.time()

                output_image=sess.run(network,feed_dict={net_input:input_image})


                output_image=np.array(output_image[0,:,:,:])
                output_image=reverse_one_hot(output_image)
                out_vis_image=colour_code_segmentation(output_image, label_values)

                accuracy, class_accuracies, prec, rec, f1, iou=evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes)

                #file_name=filepath_to_name(val_input_names[ind])
                #target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))
                #for item in class_accuracies:
                #    target.write(", %f"%(item))
                #target.write("\n")

                scores_list.append(accuracy)
                class_scores_list.append(class_accuracies)
                precision_list.append(prec)
                recall_list.append(rec)
                f1_list.append(f1)
                iou_list.append(iou)
                #tfp="%s%s/%04d/%s_pred.png"%(model_path,"checkpoints",epoch,file_name)
                #tfg="%s%s/%04d/%s_gt.png"%(model_path,"checkpoints",epoch, file_name)
                #cv2.imwrite(tfp,cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
                #cv2.imwrite(tfg,cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))


            #target.close()

            avg_score=np.mean(scores_list)
            class_avg_scores=np.mean(class_scores_list, axis=0)
            avg_scores_per_epoch.append(avg_score)
            avg_precision=np.mean(precision_list)
            avg_recall=np.mean(recall_list)
            avg_f1=np.mean(f1_list)
            avg_f1_per_epoch.append(avg_f1)
            avg_iou=np.mean(iou_list)
            avg_iou_per_epoch.append(avg_iou)

            print("\nAverage validation accuracy for epoch # %04d=%f"% (epoch, avg_score))
            print("Average per class validation accuracies for epoch # %04d:"% (epoch))
            for index, item in enumerate(class_avg_scores):
                print("%s=%f" % (class_names_list[index], item))
            print("Validation precision=", avg_precision)
            print("Validation recall=", avg_recall)
            print("Validation F1 score=", avg_f1)
            print("Validation IoU score=", avg_iou)

        epoch_time=time.time()-epoch_st
        remain_time=epoch_time*(num_epochs-1-epoch)
        m, s=divmod(remain_time, 60)
        h, m=divmod(m, 60)
        if s!=0:
            train_time="Remaining training time=%d hours %d minutes %d seconds\n"%(h,m,s)
        else:
            train_time="Remaining training time : Training completed.\n"
        LOG(train_time)
        scores_list=[]


        fig1, ax1=plt.subplots(figsize=(11, 8))

        ax1.plot(range(epoch+1-old_epoch), avg_scores_per_epoch)
        ax1.set_title("Average validation accuracy vs epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Avg. val. accuracy")


        plt.savefig('accuracy_vs_epochs.png')

        plt.clf()

        fig2, ax2=plt.subplots(figsize=(11, 8))

        ax2.plot(range(epoch-old_epoch+1), avg_loss_per_epoch)
        ax2.set_title("Average loss vs epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Current loss")

        plt.savefig('loss_vs_epochs.png')

        plt.clf()

        fig3, ax3=plt.subplots(figsize=(11, 8))

        ax3.plot(range(epoch-old_epoch+1), avg_iou_per_epoch)
        ax3.set_title("Average IoU vs epochs")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Current IoU")

        plt.savefig('iou_vs_epochs.png')
        plt.clf()

        import operator
        index, value=max(enumerate(avg_scores_per_epoch), key=operator.itemgetter(1))
        p=(index,value,model)
        import pickle
        #print("restoring model")
        model_checkpoint_name=model_path+"checkpoints/latest_model_"+ frontend + "_" + model + "_"+ ".ckpt"
        saver.restore(sess, model_checkpoint_name)
        try:
            j,m,k=pickle.load(open("%s/Frn_%s_Mod_%s_BestTR.p"%(model_path,frontend,model),'rb'))
            if((value > m) and k==model):
                print("The highest accuracy was observed for epoch %s and is %s"%(index,value))
                pickle.dump(p, open("%s/Frn_%s_Mod_%s_BestTR.p"%(model_path,frontend,model),'wb'))
                #if(index==epoch):
                #print("Saving model")
                saver.save(sess,"%s/BestFr_%s_model_%s.ckpt"%(model_path,frontend,model))

            else:
                print("The highest accuracy was observed for epoch %s and is %s"%(j,m))

        except:
            pickle.dump(p, open("%s/Frn_%s_Mod_%s_BestTR.p"%(model_path,frontend,model),'wb'))
            if(index==epoch):
                #print("Saving model")
                saver.save(sess,"%s/BestFr_%s_model_%s.ckpt"%(model_path,frontend,model))
        index, valueF=max(enumerate(avg_f1_per_epoch), key=operator.itemgetter(1))
        p=(index,valueF,model)
        import pickle
        try:
            j,m,k=pickle.load(open("%s/Frn_%s_Mod_%s_BestTR_F1.p"%(model_path,frontend,model),'rb'))
            if((valueF > m) and k==model):
                print("The highest F1 score was observed for epoch %s and is %s"%(index,valueF))
                pickle.dump(p, open("%s/Frn_%s_Mod_%s_BestTR_F1.p"%(model_path,frontend,model),'wb'))
                #if(index==epoch):
                print("Saving model")
                saver.save(sess,"%s/BestFr_%s_model_%s_F1.ckpt"%(model_path,frontend,model))

            else:
                print("The highest F1 score was observed for epoch %s and is %s"%(j,m))

        except:
            pickle.dump(p, open("%s/Frn_%s_Mod_%s_BestTR_F1.p"%(model_path,frontend,model),'wb'))
            if(index==epoch):
                print("Saving model")
                saver.save(sess,"%s/BestFr_%s_model_%s_F1.ckpt"%(model_path,frontend,model))

        if value > 0.999:
            print("Finished training...")
            break

    shutil.rmtree('models/checkpoints')
if __name__=='__main__':
    checksetup()
    #LabelMRC()
    #TrainMRC()
    PredictMRC()
    Check_results()
