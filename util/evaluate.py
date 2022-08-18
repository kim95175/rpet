import re
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
from models.rpet_pose import get_max_preds
from util.visualize import imshow_keypoints
from einops import rearrange, repeat

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

COCO_CLASSES = ('person')

palette = np.array([[255, 128, 0], 
                        [255, 153, 51], [255, 178, 102], [230, 230, 0], 
                        [255, 153, 255], [153, 204, 255], [255, 102, 255], 
                        [255, 51, 255], 
                        [102, 178, 255],
                        [51, 153, 255], 
                        [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255],
                        [255, 0, 0], [255, 255, 255]])

mpii_skeleton = [[8,9], 
            [11,12], [11, 10], [2, 1], [1, 0],
            [13, 14], [14, 15], [3, 4], [4, 5],
             [12, 2], [13, 3], [6, 2], [6, 3], [8, 12], [8,13]]

mpii_pose_link_color = palette[[ 9, 
                            0, 0, 7, 7,
                            0, 0, 7, 7, 
                            0, 0, 7, 7, 0, 0,
]]

mpii_pose_kpt_color = palette[[
    7, 7, 7, 7, 7,
    7, 7, 0, 0, 9, 
    0, 0, 0, 0, 0, 0
]]

def pckh(pred, target):
    num_joint = pred.shape[0]
    true_detect = np.zeros((4, num_joint))
    whole_count = np.zeros((4, num_joint))
    thr = [0.1, 0.2, 0.3, 0.5]
    
    #print(pred, target)
    if target[8][2] >= 0.7 and target[9][2] >= 0.7: 
        head_size = np.linalg.norm(target[8][:2] - target[9][:2])
    else:
        return true_detect, whole_count

    for j in range(num_joint):
        if target[j][2] < 0.7: # invisible
            continue
        dist = np.linalg.norm(target[j][:2] - pred[j][:2])
        for t in range(len(thr)):
            whole_count[t][j] += 1
            if dist <= thr[t] * head_size:
                true_detect[t][j] += 1

    return true_detect, whole_count

def getArea(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def getUnionAreas(boxA, boxB, interArea=None):
    area_A = getArea(boxA)
    area_B = getArea(boxB)
    
    if interArea is None:
        interArea = getIntersectionArea(boxA, boxB)
        
    return float(area_A + area_B - interArea)

def getIntersectionArea(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # intersection area
    return (xB - xA + 1) * (yB - yA + 1)

def boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    return True

def iou(boxA, boxB):
    # if boxes dont intersect
    if boxesIntersect(boxA, boxB) is False:
        return 0
    interArea = getIntersectionArea(boxA, boxB)
    union = getUnionAreas(boxA, boxB, interArea=interArea)
    
    # intersection over union
    result = interArea / union
    if result >= 0:
        return result
    else:
        return 0


def calculateAveragePrecision(rec, prec):
    mrec = [0] + [e for e in rec] + [1]
    mpre = [0] + [e for e in prec] + [0]
    #print(mrec, mpre)
    for i in range(len(mpre)-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])

    ii = []

    for i in range(len(mrec)-1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i+1)

    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i-1]) * mpre[i])
    
    return [ap, mpre[0:len(mpre)-1], mrec[0:len(mpre)-1], ii]


def ElevenPointInterpolatedAP(rec, prec):

    mrec = [e for e in rec]
    mpre = [e for e in prec]

    recallValues = np.linspace(0, 1, 11)
    recallValues = list(recallValues[::-1])
    rhoInterp, recallValid = [], []

    for r in recallValues:
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0

        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])

        recallValid.append(r)
        rhoInterp.append(pmax)

    ap = sum(rhoInterp) / 11

    return [ap, rhoInterp, recallValues, None]


# Cacluate Average Precision (AP) and visualize results
def AP(targets, results, imgs=None, 
        IOUThreshold = 0.5, method = 'AP', vis=False, 
        img_dir=None, boxThrs=0.5):
  
    if img_dir is not None:
        os.makedirs(img_dir, exist_ok=True)
    detections, groundtruths, classes = [], [], []
    batch_size = len(targets)
    result = []

    
    # ground truth data
    for i in range(batch_size):
        # Ground Truth Data
        img_size = targets[i]['orig_size']
        num_obj = targets[i]['labels'].shape[0]
        img_id = targets[i]['image_id']


        bbox = targets[i]['boxes']
        boxes = box_ops.box_cxcywh_to_xyxy(bbox)

        img_h, img_w = img_size
        boxes = torch.mul(boxes, img_h)

        if vis and imgs is not None:
            img = imgs[i]
            ori_size = img_size.int().cpu().numpy()
            gt_img = cv2.resize(img, (ori_size[1], ori_size[0]), interpolation=cv2.INTER_AREA)
            pred_img = gt_img.copy()

        for j in range(num_obj):
            label, conf = targets[i]['labels'][j].item(), 1
            x1, y1, x2, y2 = boxes[j]
            box_info = [img_id, label, conf, (x1.item(), y1.item(), x2.item(), y2.item())]
            groundtruths.append(box_info)
            if label not in classes:
                classes.append(label)
            if vis:
                x1, y1, x2, y2 = x1.int().item(), y1.int().item(), x2.int().item(), y2.int().item()
                rand_num = (img_id*x1)%19
                color= COLORS[rand_num]
                cv2.rectangle(gt_img, (x1, y1), (x2, y2), color, 1)
                label_name = COCO_CLASSES[label]
                text_str = '%s: %.2f' % (label_name, conf)
                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale, font_thickness = 0.6, 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(gt_img, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)  # draw bbox
                cv2.putText(gt_img, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA) # append label and confidence score
                cv2.putText(gt_img, 'GT', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            
        # Prediction Data
        pred = results[i]
        pred_score = pred['scores']
        pred_label = pred['labels']
        pred_boxes = pred['boxes']
        pred_indices = pred['indices']
       
        for q in range(pred_boxes.shape[0]):
            label, conf = pred_label[q].item(), pred_score[q].item()
            x1, y1, x2, y2 = pred_boxes[q]
            box_info = [img_id, label, conf, (x1.item(), y1.item(), x2.item(), y2.item())]
            index = pred_indices[q].item()
            if pred_score[q] > boxThrs:
                detections.append(box_info)
            if label not in classes:
                classes.append(label)
            if vis:
                if pred_score[q] > boxThrs:
                    x1, y1, x2, y2 = x1.int().item(), y1.int().item(), x2.int().item(), y2.int().item()
                    rand_num = (img_id*x1)%19
                    color= COLORS[rand_num]
                    cv2.rectangle(pred_img, (x1, y1), (x2, y2), color, 1)
                    label_name = COCO_CLASSES[label]
                    text_str = '%s[%d]: %.2f' % (label_name, index, conf)
                    font_face = cv2.FONT_HERSHEY_DUPLEX
                    font_scale, font_thickness = 0.6, 1

                    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
                    text_pt = (x1, y1 - 3)
                    text_color = [255, 255, 255]

                    cv2.rectangle(pred_img, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                    cv2.putText(pred_img, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
                    cv2.putText(pred_img, 'pred', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        
        if vis and i == 0:
            res = np.concatenate((pred_img, gt_img), axis=1)
            cv2.imwrite(os.path.join(img_dir, str()) +'_{}_bbox.png'.format(img_id), res)

    for c in classes:
        dects = [d for d in detections if d[1] == c]
        gts = [g for g in groundtruths if g[1] == c]

        npos = len(gts)

        dects = sorted(dects, key = lambda conf : conf[2], reverse=True)

        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))

        det = Counter(cc[0] for cc in gts)
        
    
        avg_iou = 0
        for key, val in det.items():
            det[key] = np.zeros(val)

        for d in range(len(dects)):
            gt = [gt for gt in gts if gt[0] == dects[d][0]]
            iouMax = 0

            for j in range(len(gt)):
                iou1 = iou(dects[d][3], gt[j][3])

                if iou1 > iouMax:
                    iouMax = iou1
                    jmax = j

            avg_iou += iouMax
            if iouMax >= IOUThreshold:
                if det[dects[d][0]][jmax] == 0:
                    TP[d] = 1
                    det[dects[d][0]][jmax] = 1
                else:
                    FP[d] = 1
            else:
                FP[d] = 1

        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        rec = acc_TP / npos 
        prec = np.divide(acc_TP, (acc_FP + acc_TP))

        if method == "AP":
            [ap, mpre, mrec, ii] = calculateAveragePrecision(rec, prec)
        else:
            [ap, mpre, mrec, _] = ElevenPointInterpolatedAP(rec, prec)

        det_iou = avg_iou / len(dects) if len(dects) != 0 else 0.
        r = {
            'class' : c,
            'precision' : prec,
            'recall' : rec,
            'iou' : det_iou,
            'AP' : ap,
            'interpolated precision' : mpre,
            'interpolated recall' : mrec,
            'total positives' : npos,
            'total TP' : np.sum(TP),
            'total FP' : np.sum(FP)
        }

        result.append(r)

    return result

def pose_AP(targets, results, res_pose, imgs=None,  
        IOUThreshold = 0.5, method = 'AP', vis=False, 
        img_dir=None, boxThrs=0.5, dr_size=256, pose_method='simdr'):
    
    if img_dir is not None:
        os.makedirs(img_dir, exist_ok=True)
    #print(targets, results)
    detections, groundtruths, classes = [], [], []
    batch_size = len(targets)
    result = []
    
    # ground truth data
    for i in range(batch_size):
        # Ground Truth Data
        img_size = targets[i]['orig_size']
        ori_size = img_size.int().cpu().numpy()
        num_obj = targets[i]['labels'].shape[0]
        img_id = targets[i]['image_id']
            
        bbox = targets[i]['boxes']
        boxes = box_ops.box_cxcywh_to_xyxy(bbox)
        #mask_size = targets[i]['mask_size']

        poses = targets[i]['cd']
        result_window = np.array([ori_size[0], ori_size[1], 1])
        gt_poses = np.copy(poses) * result_window

        img_h, img_w = img_size
        boxes = torch.mul(boxes, img_h)

        if vis and imgs is not None:
            img = imgs[i]
            gt_img = cv2.resize(img, (ori_size[1], ori_size[0]), interpolation=cv2.INTER_AREA)
            pred_img = gt_img.copy()
            blank_img = np.zeros((ori_size[1], ori_size[0], 3), dtype = np.uint8)
            #print("blank_img ", blank_img.shape)
           
        gt_pose_to_draw = []
        for j in range(num_obj):
            label, conf = targets[i]['labels'][j].item(), 1
            x1, y1, x2, y2 = boxes[j]

            gt_pose_to_draw.append(gt_poses[j])
            #print(gt_poses[j].shape)

            box_info = [img_id, label, conf, (x1.item(), y1.item(), x2.item(), y2.item())]
            pose_info = [img_id, gt_poses[j], None, (x1.item(), y1.item(), x2.item(), y2.item())]

            groundtruths.append(pose_info)
            if label not in classes:
                classes.append(label)
            if vis:
                x1, y1, x2, y2 = x1.int().item(), y1.int().item(), x2.int().item(), y2.int().item()
                rand_num = (img_id*x1)%19
                color= COLORS[rand_num]
                cv2.rectangle(gt_img, (x1, y1), (x2, y2), color, 1)
                label_name = COCO_CLASSES[label]
                text_str = '%s: %.2f' % (label_name, conf)
                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale, font_thickness = 0.6, 1
                
                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(gt_img, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)  # draw bbox
                cv2.putText(gt_img, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA) # append label and confidence score
               
        if vis and i == 0:
            imshow_keypoints(gt_img, gt_pose_to_draw, mpii_skeleton, kpt_score_thr=0, \
                     pose_kpt_color=mpii_pose_kpt_color, pose_link_color=mpii_pose_link_color, radius=5, thickness=2)

        # Prediction Data
        pred = results[i]
        pred_score = pred['scores']
        pred_label = pred['labels']
        pred_boxes = pred['boxes']
        pred_indices = pred['indices']

        res_pose_ = res_pose[i]#.cpu().numpy()
        if pose_method != 'hm':
            result_window = np.array([ori_size[0]/dr_size, ori_size[1]/dr_size, 1])
            res_pose_ = np.copy(res_pose_) * result_window
        pose_to_draw = []
        num_obj = 0
        for q in range(pred_boxes.shape[0]):
            label, conf = pred_label[q].item(), pred_score[q].item()
            x1, y1, x2, y2 = pred_boxes[q]
            
            index = pred_indices[q].int().item()


            box_info = [img_id, label, conf, (x1.item(), y1.item(), x2.item(), y2.item())]
            pose_info = [img_id, res_pose_[index], conf]
            if pred_score[q] > boxThrs:
                detections.append(pose_info)
            if label not in classes:
                classes.append(label)
            if vis:
                if pred_score[q] > boxThrs:
                    num_obj += 1
                    pose_to_draw.append(res_pose_[index])
                    
                    x1, y1, x2, y2 = x1.int().item(), y1.int().item(), x2.int().item(), y2.int().item()
                    rand_num = (img_id*x1)%19
                    color= COLORS[rand_num]
                    cv2.rectangle(pred_img, (x1, y1), (x2, y2), color, 1)
                    label_name = COCO_CLASSES[label]
                    text_str = '%s[%d]: %.2f' % (label_name, index, conf)
                    font_face = cv2.FONT_HERSHEY_DUPLEX
                    font_scale, font_thickness = 0.6, 1

                    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
                    text_pt = (x1, y1 - 3)
                    text_color = [255, 255, 255]

                    cv2.rectangle(pred_img, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                    cv2.putText(pred_img, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

        if vis and i == 0:

            imshow_keypoints(blank_img, pose_to_draw, mpii_skeleton, kpt_score_thr=0, \
                     pose_kpt_color=mpii_pose_kpt_color, pose_link_color=mpii_pose_link_color, radius=5, thickness=2)
            res = np.concatenate((pred_img, blank_img, gt_img), axis=1)
            cv2.imwrite(os.path.join(img_dir, str()) +'_{}_pose_num{}.png'.format(img_id, num_obj), res)

    dects = detections
    gts = groundtruths
    
    kpt_ap = []
    npos = len(gts)
    dects = sorted(dects, key = lambda conf : conf[2], reverse=True)
    for kpt in range(16):

        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))

        det = Counter(cc[0] for cc in gts)

        for key, val in det.items():
            det[key] = np.zeros(val)

        for d in range(len(dects)):
            gt = [gt for gt in gts if gt[0] == dects[d][0]]
            dist_min = 10000
            gt_head_size = 0  
            
            np_pred = dects[d][1]
                
            for j in range(len(gt)):
                np_gt = gt[j][1]
                #print(np_pred.shape, np_gt.shape)
                if np_gt[8][2] >= 0.5 and np_gt[9][2] >= 0.5: 
                    head_size = np.linalg.norm(np_gt[7][:2] - np_gt[9][:2])
                else:
                    continue
                #if np_gt[kpt][2] < 0.7: # invisible
                #    continue
                dist = np.linalg.norm(np_gt[kpt][:2] - np_pred[kpt][:2])

                if dist < dist_min:
                    dist_min = dist
                    gt_head_size = head_size 
                    jmax = j

            if dist_min <= IOUThreshold * gt_head_size:
                if det[dects[d][0]][jmax] == 0:
                    TP[d] = 1
                    det[dects[d][0]][jmax] = 1
                else:
                    FP[d] = 1
            else:
                FP[d] = 1
        
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        rec = acc_TP / npos
        prec = np.divide(acc_TP, (acc_FP + acc_TP))
        if method == "AP":
            [ap, mpre, mrec, ii] = calculateAveragePrecision(rec, prec)
        else:
            [ap, mpre, mrec, _] = ElevenPointInterpolatedAP(rec, prec)

        kpt_ap.append(ap)

    r = {
        'class' : 0,
        'AP' : kpt_ap,
    }
    result.append(r)

    return result



def mAP(result):
    ap = 0
    for r in result:
        ap += r['AP']

    #print("mAP len(result) = ", len(result))
    mAP = ap / len(result) if len(result) != 0 else 0.
    return mAP

def pck_mAP(result):
    kpt_result = np.zeros(9)
    ap = [0.] * 16
    for r in result:
        for k in range(16):
            ap[k] += r['AP'][k]
    
    if len(result) != 0:
        for k in range(16):
            ap[k] = ap[k] / len(result)
    print(ap)

    kpt_result[0] = ap[9] # HEAD
    kpt_result[1] = ap[8] # NECK
    kpt_result[2] = (ap[12] + ap[7] + ap[13]) / 3  # SHO
    kpt_result[3] = (ap[11] + ap[14]) /2 # ELB
    kpt_result[4] = (ap[10] + ap[15]) /2 # WRI
    kpt_result[5] = (ap[2] + ap[3]+ ap[6]) /3 # HIP
    kpt_result[6] = (ap[1] + ap[4]) /2 # KNE
    kpt_result[7] = (ap[0] + ap[5]) /2 # ANK
    kpt_result[8] = np.average(kpt_result[:8]) # TOT
    #print(ap)
    return kpt_result #ap

def mIOU(result):
    iou = 0
    for r in result:
        iou += r['iou']

    mIOU = iou / len(result) if len(result) != 0 else 0.
    return mIOU
