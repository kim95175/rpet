import matplotlib.pyplot as plt
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from collections import Counter
import torch
from PIL import Image
from util.misc import nested_tensor_from_tensor_list
from util import box_ops

COLORS = ((244,  67,  54),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          (  0, 150, 136),
          ( 76, 175,  80),
          (139, 195,  74),
          (205, 220,  57),
          (255, 235,  59),
          (255, 193,   7),
          (255, 152,   0),
          (255,  87,  34),
          (121,  85,  72),
          (158, 158, 158),
          ( 96, 125, 139))


import mmcv
import torch
from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow
import math


def imshow_keypoints(img,
                     pose_result,
                     skeleton=None,
                     kpt_score_thr=0.3,
                     pose_kpt_color=None,
                     pose_link_color=None,
                     radius=4,
                     thickness=1,
                     show_keypoint_weight=False):
    """Draw keypoints and links on an image.
    Args:
            img (str or Tensor): The image to draw poses on. If an image array
                is given, id will be modified in-place.
            pose_result (list[kpts]): The poses to draw. Each element kpts is
                a set of K keypoints as an Kx3 numpy.ndarray, where each
                keypoint is represented as x, y, score.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints. If None,
                the keypoint will not be drawn.
            pose_link_color (np.array[Mx3]): Color of M links. If None, the
                links will not be drawn.
            thickness (int): Thickness of lines.
    """
    img_h, img_w, _ = img.shape
    
    for kpts in pose_result:
        kpts = np.array(kpts, copy=False)
        # draw each point on image
        if pose_kpt_color is not None:
            assert len(pose_kpt_color) == len(kpts)
            for kid, kpt in enumerate(kpts):
                x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]
                #if kpt_score > kpt_score_thr:
                if show_keypoint_weight:
                    img_copy = img.copy()
                    r, g, b = pose_kpt_color[kid]
                    cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                                radius, (int(r), int(g), int(b)), -1)
                    transparency = max(0, min(1, kpt_score))
                    cv2.addWeighted(
                        img_copy,
                        transparency,
                        img,
                        1 - transparency,
                        0,
                        dst=img)
                else:
                    r, g, b = pose_kpt_color[kid]
                    if kid == 7:
                        rad = 0
                    else: rad = radius
                    #rad = radius
                    cv2.circle(img, (int(x_coord), int(y_coord)), rad,
                                (int(r), int(g), int(b)), -1)

        # draw links
        if skeleton is not None and pose_link_color is not None:
            assert len(pose_link_color) == len(skeleton)
            for sk_id, sk in enumerate(skeleton):
                pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
                if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                        and pos1[1] < img_h and pos2[0] > 0 and pos2[0] < img_w
                        and pos2[1] > 0 and pos2[1] < img_h):
                        #and kpts[sk[0], 2] > kpt_score_thr
                        #and kpts[sk[1], 2] > kpt_score_thr):
                    r, g, b = pose_link_color[sk_id]
                    if show_keypoint_weight:
                        img_copy = img.copy()
                        X = (pos1[0], pos2[0])
                        Y = (pos1[1], pos2[1])
                        mX = np.mean(X)
                        mY = np.mean(Y)
                        length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                        angle = math.degrees(
                            math.atan2(Y[0] - Y[1], X[0] - X[1]))
                        stickwidth = 2
                        polygon = cv2.ellipse2Poly(
                            (int(mX), int(mY)),
                            (int(length / 2), int(stickwidth)), int(angle), 0,
                            360, 1)
                        cv2.fillConvexPoly(img_copy, polygon,
                                           (int(r), int(g), int(b)))
                        transparency = max(
                            0, min(1, 0.5 * (kpts[sk[0], 2] + kpts[sk[1], 2])))
                        cv2.addWeighted(
                            img_copy,
                            transparency,
                            img,
                            1 - transparency,
                            0,
                            dst=img)
                    else:
                        cv2.line(
                            img,
                            pos1,
                            pos2, (int(r), int(g), int(b)),
                            thickness=thickness)
        # draw nose to neck
        #pos1 = (int(kpts[0, 0]), int(kpts[0, 1]))  # nose
        #pos2 = (int((kpts[1, 0] + kpts[2, 0])/2 ), int((kpts[1, 1] + kpts[2, 1])/2 ))
        #r, g, b = pose_kpt_color[0]
        #cv2.line(img,pos1,pos2, (int(r), int(g), int(b)) , thickness=thickness)

    return img



def show_result(img,
                size,
                result,
                skeleton=None,
                kpt_score_thr=0, #0.3,
                bbox_color=None,
                pose_kpt_color=None,
                pose_link_color=None,
                radius=4,
                thickness=1,
                font_scale=0.5,
                win_name='',
                show=False,
                show_keypoint_weight=False,
                wait_time=0,
                out_file=None):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        result (list[dict]): The results to draw over `img`
            (bbox_result, pose_result).
        skeleton (list[list]): The connection of keypoints.
            skeleton is 0-based indexing.
        kpt_score_thr (float, optional): Minimum score of keypoints
            to be shown. Default: 0.3.
        pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
            If None, do not draw keypoints.
        pose_link_color (np.array[Mx3]): Color of M links.
            If None, do not draw links.
        radius (int): Radius of circles.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        win_name (str): The window name.
        show (bool): Whether to show the image. Default: False.
        show_keypoint_weight (bool): Whether to change the transparency
            using the predicted confidence scores of keypoints.
        wait_time (int): Value of waitKey param.
            Default: 0.
        out_file (str or None): The filename to write the image.
            Default: None.

    Returns:
        Tensor: Visualized image only if not `show` or `out_file`
    """
    img = mmcv.imread(img)
    img = img.copy()
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    img_h, img_w, _ = img.shape

    pose_result = []
    for res in result: #result shape should be (people_num, keypoint_num, 3(x,y,prob))
        pose_result.append(res)

    imshow_keypoints(img, pose_result, skeleton, kpt_score_thr,
                     pose_kpt_color, pose_link_color, radius, thickness)

    if show:
        imshow(img, win_name, wait_time)

    if out_file is not None:
        imwrite(img, out_file)

    return img

def vis_pose_result(img,
                    size,
                    result,
                    radius=4,
                    thickness=1,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    show=False,
                    out_file=None):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str | np.ndarray): Image filename or loaded image.
        result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
        radius (int): Radius of circles.
        thickness (int): Thickness of lines.
        kpt_score_thr (float): The threshold to visualize the keypoints.
        skeleton (list[tuple()]): Default None.
        show (bool):  Whether to show the image. Default True.
        out_file (str|None): The filename of the output visualization image.
    """

    # TODO: These will be removed in the later versions.
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255],
                        [255, 0, 0], [255, 255, 255]])

    skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                [3, 5], [4, 6]]

    pose_link_color = palette[[
        0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
    ]]
    pose_kpt_color = palette[[
        16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
    ]]

    result_window = np.array([2, 2, 1])
    #result_window = np.array([size, size, 1])
    result = np.copy(result) *result_window


    img = show_result(
        img,
        size,
        result,
        skeleton,
        radius=radius,
        thickness=thickness,
        pose_kpt_color=pose_kpt_color,
        pose_link_color=pose_link_color,
        kpt_score_thr=kpt_score_thr,
        bbox_color=bbox_color,
        show=show,
        out_file=out_file)

    return img

def save_batch_heatmaps(batch_image, batch_heatmaps, targets, img_dir, results,
                        normalize=False):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if img_dir is not None:
        os.makedirs(img_dir, exist_ok=True)
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    #print(batch_heatmaps.shape)
    batch_size = batch_heatmaps.size(0)
    num_quries = batch_heatmaps.size(1)
    num_joints = batch_heatmaps.size(2)
    heatmap_height = 256 #512 #128#batch_heatmaps.size(2)
    heatmap_width = 256 #512 #128#batch_heatmaps.size(3)

    img_size = 128
    for i in range(batch_size):
        if i != 0:
            continue

        ids = targets[i]['image_id']
        pred_score = results[i]['scores']
        image = batch_image[i]
        query_heatmaps = batch_heatmaps[i]

        gt_heatmaps = targets[i]['hm']
        num_gt = gt_heatmaps.shape[0]
        for g in range(num_gt):
            grid_image = np.zeros((heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)
            
            heatmaps = gt_heatmaps[g]
            heatmaps = heatmaps.clone()
            min = float(heatmaps.min())
            max = float(heatmaps.max())
            heatmaps.add_(-min).div_(max - min + 1e-5)
                                
            heatmaps = heatmaps.mul(255)\
                                        .clamp(0, 255)\
                                        .byte()\
                                        .cpu().numpy()
            resized_image = cv2.resize(image,
                                    (int(heatmap_width), int(heatmap_height)), interpolation=cv2.INTER_AREA)

            
            height_begin = heatmap_height * i
            height_end = heatmap_height * (i + 1)
            for j in range(num_joints):
                heatmap = heatmaps[j, :, :]
                heatmap = cv2.resize(heatmap, (int(heatmap_width), int(heatmap_height)),
                                    interpolation=cv2.INTER_AREA)
                
                colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                masked_image = colored_heatmap*0.7 + resized_image*0.3
                width_begin = heatmap_width * (j+1)
                width_end = heatmap_width * (j+2)
                grid_image[height_begin:height_end, width_begin:width_end, :] = \
                    masked_image
                
            grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image
            
        for q in range(num_quries):
            if pred_score[q] < 0.8:
                continue
            grid_image = np.zeros((heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)
            
            heatmaps = query_heatmaps[q]
            heatmaps = heatmaps.clone()
            min = float(heatmaps.min())
            max = float(heatmaps.max())
            heatmaps.add_(-min).div_(max - min + 1e-5)

            heatmaps = heatmaps.mul(255)\
                                        .clamp(0, 255)\
                                        .byte()\
                                        .cpu().numpy()

            resized_image = cv2.resize(image,
                                    (int(heatmap_width), int(heatmap_height)), interpolation=cv2.INTER_AREA)
            
            height_begin = heatmap_height * i
            height_end = heatmap_height * (i + 1)
            masked_acc_image = resized_image*0.2
            for j in range(num_joints):
                heatmap = heatmaps[j, :, :]
                heatmap = cv2.resize(heatmap, (int(heatmap_width), int(heatmap_height)),
                                    interpolation=cv2.INTER_LINEAR)#cv2.INTER_AREA)

                colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)#cv2.COLORMAP_JET)
                masked_image = colored_heatmap*0.7 + resized_image*0.3

                masked_acc_image += colored_heatmap*0.2

                width_begin = heatmap_width * (j+1)
                width_end = heatmap_width * (j+2)
                grid_image[height_begin:height_end, width_begin:width_end, :] = \
                    masked_image


            grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image
            cv2.imwrite(os.path.join(img_dir, str()) +'_{}_heatamp_pose_q{}.png'.format(ids,q),  grid_image)
            masked_image = np.clip(masked_image, 0, 255)
            cv2.imwrite(os.path.join(img_dir, str()) +'_{}_heatamp_pose_acc_q{}.png'.format(ids,q),  masked_acc_image)


