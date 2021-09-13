from __future__ import print_function, division
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import numpy as np
# import cv2
from util import *
from darknet import Darknet
# from preprocess import prep_image, inp_to_image
# import pandas as pd
import random
import argparse
import pickle as pkl
import json
from datetime import datetime


Focus_length = 28


def get_test_input(input_dim, CUDA):
    img = cv2.imread("imgs/messi.jpg")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def write(x, img):
    c1 = tuple(x[1:3].cpu().numpy().tolist())
    c2 = tuple(x[3:5].cpu().numpy().tolist())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), color, -1)
    cv2.putText(img, label, (int(c1[0]), int(c1[1]) + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

    if int(c2[1]) - int(c1[1]) == 0:
        pass
    else:
        distance = "{:.2f}".format(float(29 * Focus_length * 1.5 / (int(c2[1]) - int(c1[1]))))
        t_size_1 = cv2.getTextSize(distance, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        # right-top corner
        c_r = tuple([int(c2[0]), int(c1[1])])
        c4 = c_r[0] + t_size_1[0] + 4, c_r[1] + t_size_1[1] + 5
        cv2.rectangle(img, c_r, (int(c4[0]), int(c4[1])), color, -1)
        if 29 * Focus_length * 1.5 / (int(c2[1]) - int(c1[1])) <= 20:
            cv2.putText(img, distance, (c_r[0], c_r[1] + t_size_1[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [225, 0, 0], 1)
        else:
            cv2.putText(img, distance, (c_r[0], c_r[1] + t_size_1[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 225, 225], 1)
    return img


def exist_item(x):
    c1 = tuple(x[1:3].cpu().numpy().tolist())
    c2 = tuple(x[3:5].cpu().numpy().tolist())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])

    if int(c2[1]) - int(c1[1]) == 0:
        distance = 1000
    else:
        distance = "{:.2f}".format(float(29 * Focus_length * 1.5 / (int(c2[1]) - int(c1[1]))))

    return label, distance


def arg_parse():
    """
    Parse arguements to the detect module
    
    """

    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.25)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="160", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    # os.system("gdrive mkdir detection_results")
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()

    num_classes = 80
    bbox_attrs = 5 + num_classes

    model = Darknet(cfgfile)
    model.load_weights(weightsfile)

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])

    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model.eval()
    cap = cv2.VideoCapture(0)

    assert cap.isOpened(), 'Cannot capture source'

    detection_results = []
    frames = 0
    start = time.time()
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:

            img, orig_im, dim = prep_image(frame, inp_dim)
            im_dim = torch.FloatTensor(dim).repeat(1, 2)

            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim)) / inp_dim

            #            im_dim = im_dim.repeat(output.size(0), 1)
            output[:, [1, 3]] *= frame.shape[1]
            output[:, [2, 4]] *= frame.shape[0]

            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))

            list(map(lambda x: write(x, orig_im), output))

            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))

            if frames % 90 == 0:
                detection_results.append([datetime.now(), "{:5.2f}".format(frames / (time.time() - start)),
                                          list(map(lambda x: exist_item(x), output))])
                with open('detection_results/' + 'detection_results' + str(frames) + '.json', 'w') as outfile:
                    json.dump(detection_results, outfile, indent=4, sort_keys=True, default=str)

                cmd = "gdrive upload --parent 1LuL43ZBOk3GnRMkn_CxKR1dxBROgDBEt {}".format(
                    'detection_results/' + 'detection_results' + str(frames) + '.json')
                os.system(cmd)

        else:
            break
