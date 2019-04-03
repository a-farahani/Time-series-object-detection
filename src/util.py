#!/usr/bin/python
# -*- coding: utf-8 -*-
import random
import json
import time
import glob
import os
import re

import numpy as np
import image_slicer
import cv2


def combine(input_path, output_path):
    """
    create an image per video by taking average of all frames.

    Parameters
    ----------
    input_path : string
        path to the dataset
    output_path : string
        path to the output directory
    """

    # output_dict = {}
    video_dirs = os.listdir(input_path)
    for video_dir in video_dirs:
        img_path = sorted(glob.glob(os.path.join(input_path, video_dir) +
                                    '/images/*'))

        average = cv2.imread(img_path[0], 0).astype(np.float)
        for img_file in img_path[1:]:
            img = cv2.imread(img_file, 0)
            average += img

        average /= len(img_path)
        average = cv2.normalize(average, None, 0, 255, cv2.NORM_MINMAX)
        if 'test' not in video_dir:
            cv2.imwrite(output_path + '/train/data/' + video_dir +
                        '.png', average)
        else:
            cv2.imwrite(output_path + '/test/data/' + video_dir +
                        '.png', average)


def json_to_mask(input_path, output_path):
    """
    creates masks from json files

    Parameters
    ----------
    input_path : string
        path to the dataset
    output_path : string
        path to save mask files
    Raises
    ------
    OSError if the output path does not exist.
    """
    video_dirs = os.listdir(input_path)

    for video_dir in video_dirs:
        img_path = os.listdir(os.path.join(input_path, video_dir) + '/images/')

        # crate mask for train data only
        if 'test' not in video_dir:
            avg_image = cv2.imread(os.path.join(input_path, video_dir) +
                                   '/images/' + img_path[0])
            mask = np.zeros_like(avg_image)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = np.transpose(mask)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            json_path = os.path.join(input_path, video_dir) \
                + '/regions/regions.json'
            with open(json_path) as fp:
                obj = json.load(fp)

                for cnt in obj:
                    coor = cnt['coordinates']
                    contour = np.asarray([[np.asarray(c)] for c in
                                          coor], dtype='int32')
                    contour.astype('int32')
                    cv2.drawContours(mask, contour, -1, (0, 255, 0),
                                     thickness=-1)

                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask = np.transpose(mask)
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                mask = mask / 255
                try:
                    cv2.imwrite(output_path + '/train/masks/' +
                                video_dir + '.png', mask)
                except OSError:
                    raise OSError('output directory does not exist!')


def slicer(path):
    """
    Increases the number of images by slicing.

    Parameters
    ----------
    path : string
        specifies the working directory for reading and writing
        images and their masks.
    """

    img_path = path + '/train/data/'
    msk_path = path + '/train/masks/'

    img_files = os.listdir(img_path)
    mask_files = os.listdir(msk_path)

    for img_f in img_files:
        image_slicer.slice(os.path.join(img_path, img_f), 4)

    for mask_f in mask_files:
        image_slicer.slice(os.path.join(msk_path, mask_f), 4)


def rotate(path):
    """
    Increases the number of images by rotation.

    Parameters
    ----------
    path : string
        specifies the working directory for reading and writing
        images and their masks

    Returns
    -------
    None
    """

    img_path = path + '/train/data/'
    msk_path = path + '/train/masks/'

    img_files = os.listdir(img_path)
    mask_files = os.listdir(msk_path)

    for img_f in img_files:
        img = cv2.imread(os.path.join(img_path, img_f))
        (row, col, _) = img.shape

        for i in range(1, 4):
            rot_mat = cv2.getRotationMatrix2D((col / 2, row / 2), 90, 1)
            img = cv2.warpAffine(img, rot_mat, (col, row))

            cv2.imwrite(img_path + img_f[:-4] + '_rot_' + str(i * 90) +
                        '.png', img)

    for mask_f in mask_files:
        mask = cv2.imread(msk_path + mask_f)
        (row_m, col_m, _) = mask.shape

        for i in range(1, 4):
            rotm_mat = cv2.getRotationMatrix2D((col_m / 2, row_m / 2),
                                               90, 1)
            mask = cv2.warpAffine(mask, rotm_mat, (col_m, row_m))
            cv2.imwrite(msk_path + mask_f[:-4] + '_rot_' + str(i * 90) +
                        '.png', mask)


def transpose(path):
    """
    Increases the number of images by transposing.

    Parameters
    ----------
    path : string
        specifies the working directory for reading and writing
        images and their masks

    Returns
    -------
    None
    """

    img_path = path + '/train/data/'
    msk_path = path + '/train/masks/'

    img_files = os.listdir(img_path)
    mask_files = os.listdir(msk_path)

    for img_f in img_files:
        img = cv2.imread(os.path.join(img_path, img_f), 0)
        img = np.transpose(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(img_path + img_f[:-4] + '_tr.png', img)

    for mask_f in mask_files:
        mask = cv2.imread(msk_path + mask_f, 0)
        mask = np.transpose(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(msk_path + mask_f[:-4] + '_tr.png', mask)


def split(path, ratio=0.2):
    """
    splits training set

    Parameters
    ----------
    path : string
        path to the output directory
    ratio : float
        specifies the size of validation set

    Returns
    -------
    None
    """

    train_data = path + '/train/data/'
    train_msk = path + '/train/masks/'

    val_data = path + '/validate/data/'
    val_msk = path + '/validate/masks/'

    img_files = os.listdir(train_data)

    val_size = int(ratio * len(img_files))
    val_files = random.sample(img_files, val_size)

    for val_file in val_files:
        os.rename(os.path.join(train_data, val_file),
                  os.path.join(val_data, val_file))
        os.rename(os.path.join(train_msk, val_file),
                  os.path.join(val_msk, val_file))


def mask_to_json(mask_path, json_path):
    """
    converts mask files generated by Tiramisu to json file

    Parameters
    ----------
    mask_path : string
        path to mask files generated by Tiramisu
    json_path : string
        path to output directory where json files will be saved.

    Returns
    -------
    None
    """

    msk_files = os.listdir(mask_path)
    out_list = []
    for file in msk_files:
        (_, dataset_name) = os.path.split(file)
        remove = ['neurofinder.', '_msk.png', '.png']
        dataset_name = re.sub(r'|'.join(map(re.escape, remove)), '',
                              dataset_name)

        mask = cv2.imread(os.path.join(mask_path, file), 0)
        (cnt, _) = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)

        regions = []
        for c in cnt:
            c = np.reshape(c, (c.shape[0], c.shape[2])).tolist()
            regions.append({'coordinates': c})

        out_list.append({'dataset': dataset_name, 'regions': regions})

    time_stamp = time.strftime('%m-%d-%H-%M-%S', time.gmtime())
    with open(json_path + time_stamp + '.json', 'w') as fp:
        json.dump(out_list, fp)
