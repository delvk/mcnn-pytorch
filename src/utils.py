import cv2
import numpy as np
import os
import torch


def save_results(input_img, gt_data, density_map, output_dir, fname="results.png"):
    input_img = input_img[0][0]
    label_gt = "{:.2f}".format(np.sum(gt_data))
    label_et = "{:.2f}".format(np.sum(density_map))

    gt_data = 255 * gt_data / np.max(gt_data)
    density_map = 255 * density_map / np.max(density_map)
    gt_data = gt_data[0][0]
    density_map = density_map[0][0]
    if density_map.shape[1] != input_img.shape[1]:
        density_map = cv2.resize(density_map, (input_img.shape[1], input_img.shape[0]))
        gt_data = cv2.resize(gt_data, (input_img.shape[1], input_img.shape[0]))

    input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
    # ground-truth density
    r_channel = np.zeros(gt_data.shape, dtype=gt_data.dtype)
    g_channel = np.zeros(gt_data.shape, dtype=gt_data.dtype)
    b_channel = gt_data
    gt_data = cv2.merge((b_channel, g_channel, r_channel))
    result_gt_img = input_img + gt_data

    cv2.putText(
        result_gt_img, label_gt, (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2
    )

    # estimate density
    b_channel = np.zeros(density_map.shape, dtype=gt_data.dtype)
    g_channel = np.zeros(density_map.shape, dtype=gt_data.dtype)
    r_channel = density_map
    density_map = cv2.merge((b_channel, g_channel, r_channel))
    result_den_img = input_img + density_map

    cv2.putText(
        result_den_img, label_et, (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2
    )

    # stacking the two
    result_img = np.hstack((result_gt_img, result_den_img))

    cv2.imwrite(os.path.join(output_dir, fname), result_img)


def save_density_map(density_map, output_dir, fname="results.png"):
    density_map = 255 * density_map / np.max(density_map)
    density_map = density_map[0][0]
    cv2.imwrite(os.path.join(output_dir, fname), density_map)


def display_results(input_img, gt_data, density_map):
    input_img = input_img[0][0]
    max_gt = np.max(gt_data)
    if max_gt != 0:
        gt_data = 255 * gt_data / max_gt
    else:
        gt_data = 0
    density_map = 255 * density_map / np.max(density_map)
    gt_data = gt_data[0][0]
    density_map = density_map[0][0]
    if density_map.shape[1] != input_img.shape[1]:
        input_img = cv2.resize(input_img, (density_map.shape[1], density_map.shape[0]))
    result_img = np.hstack((input_img, gt_data, density_map))
    result_img = result_img.astype(np.uint8, copy=False)
    cv2.imshow("Result", result_img)
    cv2.waitKey(0)


def display_results_alt(input_img, density_map):
    label_et = "{:.2f}".format(np.sum(density_map))

    input_img = input_img[0][0]
    input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)

    density_map = 255 * density_map / np.max(density_map)
    density_map = density_map[0][0]
    density_map = density_map.astype(np.uint8, copy=False)

    if density_map.shape[1] != input_img.shape[1]:
        density_map = cv2.resize(density_map, (input_img.shape[1], input_img.shape[0]))

    cv2.putText(
        density_map, label_et, (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2
    )

    # rbg
    b_channel = np.zeros(density_map.shape, dtype=density_map.dtype)
    g_channel = np.zeros(density_map.shape, dtype=density_map.dtype)
    r_channel = density_map
    density_map = cv2.merge((b_channel, g_channel, r_channel))
    result_den_img = input_img + density_map

    cv2.putText(
        result_den_img, label_et, (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2
    )

    # stacking the two
    result_img = np.hstack((density_map, result_den_img))
    cv2.imwrite("result.png", result_img)
    # cv2.waitKey(0)


def load_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint["epoch"])
        )
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch
