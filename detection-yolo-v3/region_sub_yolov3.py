from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random
import pickle as pkl
import argparse, os
from nsdi.ndsi_model import NSDIModel
from keras.preprocessing import image
import tensorflow as tf
from statistics import mode
import subprocess


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


def resizeAndpad(img, size, padColor=0):
    h, w = img.shape[:2]
    sh, sw = size
    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA
    else:  # stretching image
        interp = cv2.INTER_CUBIC
    # aspect ratio of image
    aspect = w / h  # if on Python 2, you might need to cast as a float: float(w)/h
    # compute scaling and pad sizing
    if aspect > 1:  # horizontal image
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:  # vertical image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:  # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor,
                                              (list, tuple,
                                               np.ndarray)):  # color image but only one color provided
        padColor = [padColor] * 3
    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                    borderType=cv2.BORDER_CONSTANT,
                                    value=padColor)
    return scaled_img


def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
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
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')

    parser.add_argument("--video", dest='video', help=
    "Video to run detection upon",
                        default="/home/dong/Dataset/DeepEdge/traffic/video_clips/CAM01.mp4", type=str)
    parser.add_argument("--dataset", dest="dataset", help="Dataset on which the network has been trained",
                        default="pascal")
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    return parser.parse_args()


def preprocess_input(x, image_size=128):
    x = resizeAndpad(x, (image_size, image_size))
    return (x / 255. - 0.5) * 2  # from [0, 255] to [-1,+1]


def main():

    camera = 'CAM32'
    src = '/home/dong/Dataset/DeepEdge/Campus_video_clips/' + camera + '.avi'
    dst = '/home/dong/Dataset/DeepEdge/Campus/' + camera + '/'
    os.makedirs(dst, exist_ok=True)

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)

    CUDA = torch.cuda.is_available()

    num_classes = 80  # number of classes of YOLO model

    CUDA = torch.cuda.is_available()

    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model.eval()

    cap = cv2.VideoCapture(src)

    assert cap.isOpened(), 'Cannot capture source'

    frames = 0

    # define the Inception model
    m_128 = NSDIModel(128, 11, 'inception', feed_mid=True)
    saver = tf.train.Saver(var_list=m_128.vars_to_restore)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        # Restore graph structure
        saver.restore(sess, "/home/dong/PycharmProjects/DeepEdge/nsdi/ckpt/EE/inception-252000")
        print("Inception Model restored.")

        while cap.isOpened():

            frames += 1
            ret, frame = cap.read()

            if frames % 1000 == 0:
                print("frames = ", frames)

            if ret:

                img, orig_im, dim = prep_image(frame, inp_dim)

                im_dim = torch.FloatTensor(dim).repeat(1, 2)

                if CUDA:
                    im_dim = im_dim.cuda()
                    img = img.cuda()

                with torch.no_grad():
                    output = model(Variable(img), CUDA)
                output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

                if type(output) == int:
                    continue

                im_dim = im_dim.repeat(output.size(0), 1)
                scaling_factor = torch.min(inp_dim / im_dim, 1)[0].view(-1, 1)

                output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
                output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

                output[:, 1:5] /= scaling_factor
                output = sorted(output, key=lambda x: x[-2], reverse=True)  # output is a list now

                n = 0  # the # objects in the frame
                for i in range(len(output)):
                    if int(output[i][-1]) in [0, 1, 2, 3, 5, 7]:
                        output[i][[1, 3]] = torch.clamp(output[i][[1, 3]], 0.0, im_dim[i, 0])
                        output[i][[2, 4]] = torch.clamp(output[i][[2, 4]], 0.0, im_dim[i, 1])

                        # use to choose the region of videos, like cropping images
                        if int(output[i][4] <= 1 / 2 * im_dim[i, 1]):
                            continue

                        # read the coordinate from output
                        left = max(0, np.floor(output[i][1] + 0.5))  # xmin
                        top = max(0, np.floor(output[i][2] + 0.5))  # ymin
                        right = min(frame.shape[1], np.floor(output[i][3] + 0.5))  # xmax
                        bottom = min(frame.shape[0], np.floor(output[i][4] + 0.5))  # ymax

                        # extend coordinate to 15% larger than the original one
                        y_extend_len = (bottom - top) * 0.15
                        x_extend_len = (right - left) * 0.15
                        left = int(max(0, left - x_extend_len))
                        right = int(min(frame.shape[1], right + x_extend_len))
                        top = int(max(0, top - y_extend_len))
                        bottom = int(min(frame.shape[0], bottom + y_extend_len))
                        crop_img_np = orig_im[top:bottom, left:right, :]
                        crop_img_np = resizeAndpad(crop_img_np, (299, 299))

                        # predict the cropped image by Inception V3
                        img_np = preprocess_input(image.img_to_array(crop_img_np))

                        logits_np_list = sess.run(m_128.ee_logits_list,
                                                  feed_dict={m_128.inputs_ph: np.expand_dims(img_np, axis=0),
                                                             m_128.training_ph: False})
                        logits_end = sess.run(m_128.logits,
                                              feed_dict={m_128.inputs_ph: np.expand_dims(img_np, axis=0),
                                                         m_128.training_ph: False})

                        # rename the image name
                        a = np.argmax(softmax(np.array(logits_np_list)), axis=1)[-3:]
                        b = np.argmax(softmax(logits_end), axis=1)[0]

                        if a[0] != a[1] and a[0] != a[2] and a[1] != a[2]:
                            gt = a[np.random.randint(3)]
                        else:
                            gt = mode([a[0], a[1], a[2]])

                        # saving the cropped images to the disk, Label_groundtruth + label_prediction
                        img_dst = dst + camera + '_%s' % format(frames, '05d') + \
                                  '_%s' % format(n, '02d') + '_%s' % format(gt, '02d') + '_%s' % format(b,
                                                                                                        '02d') + '.jpg'
                        cv2.imwrite(img_dst, crop_img_np)

                        n += 1
                    else:
                        continue

            else:
                break


def control_dataset():
    # producing control dataset using the MIOTCD dataset
    eval_images_folder = '/home/dong/Dataset/DeepEdge/MIOTCD/new_val'
    image_size = 128
    model_name = 'inception'

    imlist = []
    print("Producing image lists...")
    for root, dirs, files in os.walk(eval_images_folder, topdown=False):
        for idx, file in enumerate(files):
            file_name = os.path.join(root, file)
            imlist.append(file_name)

    # define the Inception model
    m_128 = NSDIModel(128, 11, 'inception', feed_mid=True)
    saver = tf.train.Saver(var_list=m_128.vars_to_restore)

    for loop in range(1, 9):
        # folder for saving images
        camera = 'CAM' + '%s' % format(loop, '02d')
        dst = '/home/dong/Dataset/DeepEdge/Control/' + camera + '/'
        os.makedirs(dst, exist_ok=True)

        print()
        print("producing the camera:", camera)
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            # Restore graph structure
            saver.restore(sess, "/home/dong/PycharmProjects/DeepEdge/nsdi/ckpt/EE/inception-252000")
            print("Inception Model restored.")

            frames = 0
            len_imlist = len(imlist)
            for se in range(18000):

                frames += 1
                obj_per_frame = np.random.randint(1, 4, 1)

                n = 0
                for t in range(obj_per_frame[0]):
                    img_number = np.random.randint(0, len_imlist - 1, 1)
                    frame = image.load_img(imlist[img_number[0]], target_size=(image_size, image_size))
                    # predict the cropped image by Inception V3
                    img_np = preprocess_input(image.img_to_array(frame))

                    logits_np_list = sess.run(m_128.ee_logits_list,
                                              feed_dict={m_128.inputs_ph: np.expand_dims(img_np, axis=0),
                                                         m_128.training_ph: False})
                    logits_end = sess.run(m_128.logits,
                                          feed_dict={m_128.inputs_ph: np.expand_dims(img_np, axis=0),
                                                     m_128.training_ph: False})

                    # rename the image name
                    a = np.argmax(softmax(np.array(logits_np_list)), axis=1)[-3:]
                    b = np.argmax(softmax(logits_end), axis=1)[0]

                    if a[0] != a[1] and a[0] != a[2] and a[1] != a[2]:
                        gt = a[np.random.randint(3)]
                    else:
                        gt = mode([a[0], a[1], a[2]])

                    # saving the cropped images to the disk, Label_groundtruth + label_prediction
                    img_dst = dst + camera + '_%s' % format(frames, '05d') + \
                              '_%s' % format(n, '02d') + '_%s' % format(gt, '02d') + '_%s' % format(b,
                                                                                                    '02d') + '.jpg'
                    cv2.imwrite(img_dst, np.array(frame))

                    n += 1


def control_dataset_1():
    # producing control dataset using the MIOTCD dataset, each frame at least has 5 objects
    eval_images_folder = '/home/dong/Dataset/DeepEdge/MIOTCD/new_val'
    image_size = 128
    model_name = 'inception'

    imlist = []
    print("Producing image lists...")
    for root, dirs, files in os.walk(eval_images_folder, topdown=False):
        for idx, file in enumerate(files):
            file_name = os.path.join(root, file)
            imlist.append(file_name)

    # define the Inception model
    m_128 = NSDIModel(128, 11, 'inception', feed_mid=True)
    saver = tf.train.Saver(var_list=m_128.vars_to_restore)

    for loop in range(1, 5):
        # folder for saving images
        camera = 'CAM' + '%s' % format(loop, '02d')
        dst = '/home/dong/Dataset/DeepEdge/Control/' + camera + '/'
        os.makedirs(dst, exist_ok=True)

        print()
        print("producing the camera:", camera)
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            # Restore graph structure
            saver.restore(sess, "/home/dong/PycharmProjects/DeepEdge/nsdi/ckpt/EE/inception-252000")
            print("Inception Model restored.")

            frames = 0
            len_imlist = len(imlist)
            for se in range(18000):

                frames += 1
                # obj_per_frame = np.random.randint(1, 4, 1)
                obj_per_frame = [5]

                n = 0
                for t in range(obj_per_frame[0]):
                    img_number = np.random.randint(0, len_imlist - 1, 1)
                    frame = image.load_img(imlist[img_number[0]], target_size=(image_size, image_size))
                    # predict the cropped image by Inception V3
                    img_np = preprocess_input(image.img_to_array(frame))

                    logits_np_list = sess.run(m_128.ee_logits_list,
                                              feed_dict={m_128.inputs_ph: np.expand_dims(img_np, axis=0),
                                                         m_128.training_ph: False})
                    logits_end = sess.run(m_128.logits,
                                          feed_dict={m_128.inputs_ph: np.expand_dims(img_np, axis=0),
                                                     m_128.training_ph: False})

                    # rename the image name
                    a = np.argmax(softmax(np.array(logits_np_list)), axis=1)[-3:]
                    b = np.argmax(softmax(logits_end), axis=1)[0]

                    if a[0] != a[1] and a[0] != a[2] and a[1] != a[2]:
                        gt = a[np.random.randint(3)]
                    else:
                        gt = mode([a[0], a[1], a[2]])

                    # saving the cropped images to the disk, Label_groundtruth + label_prediction
                    img_dst = dst + camera + '_%s' % format(frames, '05d') + \
                              '_%s' % format(n, '02d') + '_%s' % format(gt, '02d') + '_%s' % format(b,
                                                                                                    '02d') + '.jpg'
                    cv2.imwrite(img_dst, np.array(frame))

                    n += 1


def sample_video():

    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

    src = "/home/dong/Dataset/SIMD/video_stream/person/XX/XX_4.avi"
    dsr = "../CAM32.avi"
    length = 10  # length of video sample, unit = mins
    start_time = 55 * 60  # start time , unit = mins
    end_time = start_time + length/60*3600
    ffmpeg_extract_subclip(src, start_time, end_time, targetname=dsr)


if __name__ == '__main__':
    # sample_video()
    main()
    # control_dataset()
    # control_dataset_1()

