# -*- coding: utf-8 -*-
#Usage: Extracted the faces of all the images in the DIR_PATH and save it to tmp/
#Ref :https://github.com/xiangrufan/keras-mtcnn

import tools
import os
import cv2
import numpy as np
import time
from MTCNN import create_Kao_Onet, create_Kao_Rnet, create_Kao_Pnet

#The threshold can be modified for custom usage
#Example: For the original high-accuracy usage :threshold = [0.6,0.6,0.7]
threshold = [0.4,0.4,0.5]  
DIR_PATH = 'path_to_imagedir/image_dir/'

Pnet = create_Kao_Pnet('./weight/12net.h5')
Rnet = create_Kao_Rnet('./weight/24net.h5')
Onet = create_Kao_Onet('./weight/48net.h5')  # will not work. caffe and TF incompatible


def detectFace(img, threshold):

    caffe_img = (img.copy() - 127.5) / 127.5
    origin_h, origin_w, ch = caffe_img.shape
    scales = tools.calculateScales(img)
    out = []
    t0 = time.time()
    # del scales[:4]

    for scale in scales:
        hs = int(origin_h * scale)
        ws = int(origin_w * scale)
        scale_img = cv2.resize(caffe_img, (ws, hs))
        input = scale_img.reshape(1, *scale_img.shape)
        ouput = Pnet.predict(input)  # .transpose(0,2,1,3) should add, but seems after process is wrong then.
        out.append(ouput)
    image_num = len(scales)
    rectangles = []
    for i in range(image_num):
        cls_prob = out[i][0][0][:, :,
                   1]  # i = #scale, first 0 select cls score, second 0 = batchnum, alway=0. 1 one hot repr
        roi = out[i][1][0]
        out_h, out_w = cls_prob.shape
        out_side = max(out_h, out_w)
        # print('calculating img scale #:', i)
        cls_prob = np.swapaxes(cls_prob, 0, 1)
        roi = np.swapaxes(roi, 0, 2)
        rectangle = tools.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
        rectangles.extend(rectangle)
    rectangles = tools.NMS(rectangles, 0.7, 'iou')

    t1 = time.time()
    print ('time for 12 net is: ', t1-t0)

    if len(rectangles) == 0:
        return rectangles

    crop_number = 0
    out = []
    predict_24_batch = []
    for rectangle in rectangles:
        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img, (24, 24))
        predict_24_batch.append(scale_img)
        crop_number += 1

    predict_24_batch = np.array(predict_24_batch)

    out = Rnet.predict(predict_24_batch)

    cls_prob = out[0]  # first 0 is to select cls, second batch number, always =0
    cls_prob = np.array(cls_prob)  # convert to numpy
    roi_prob = out[1]  # first 0 is to select roi, second batch number, always =0
    roi_prob = np.array(roi_prob)
    rectangles = tools.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])
    t2 = time.time()
    print ('time for 24 net is: ', t2-t1)


    if len(rectangles) == 0:
        return rectangles


    crop_number = 0
    predict_batch = []
    for rectangle in rectangles:
        # print('calculating net 48 crop_number:', crop_number)
        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img, (48, 48))
        predict_batch.append(scale_img)
        crop_number += 1

    predict_batch = np.array(predict_batch)

    output = Onet.predict(predict_batch)
    cls_prob = output[0]
    roi_prob = output[1]
    pts_prob = output[2]  # index
    # rectangles = tools.filter_face_48net_newdef(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h,
    #                                             threshold[2])
    rectangles = tools.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])
    t3 = time.time()
    print ('time for 48 net is: ', t3-t2)

    return rectangles


if __name__ == '__main__':
    os.chdir(DIR_PATH)
    if not os.path.isdir('tmp'):
        os.mkdir('tmp')
        
    for item in os.listdir('.'):
        if os.path.isfile(os.path.join('.', item)):
            if item.lower().endswith(('.png', '.jpg', '.jpeg')):
                img= cv2.imread(item)
                if img != None:
                    rec = detectFace(img,threshold)
                    faces = tools.extract_faces(img,rec)
                    for i in range(0,len(faces)):
                        cv2.imwrite('./tmp/'+str(i)+'_'+item,faces[i])

