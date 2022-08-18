# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Person Detection Network and criterion classes.
"""
from sqlite3 import adapters
from numpy.core.shape_base import stack
import torch
import torch.nn.functional as F
from torch import nn
torch.backends.cudnn.benchmark = True
from einops.einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .convnext import build_convbackbone, LayerNorm
from .matcher import build_matcher

from .rpet_pose import (RPETpose, PostProcessPoseDR, PostProcessPoseHM, NMTNORMCritierion, filter_target_simdr)
from .transformer import build_transformer



class RPET(nn.Module):
    """ This is the Person Detection Network module that performs perrson detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, 
                    num_txrx=8, aux_loss=False, box_feature='x', signal_ratio=0.75):

        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        print("transformer.d_model = ", hidden_dim)
        
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.signal_ratio=signal_ratio
        self.hidden_dim = hidden_dim
        self.input_proj_dim = hidden_dim if box_feature =='x' else int(hidden_dim * self.signal_ratio)#75)
        self.input_proj = nn.Sequential(
                LayerNorm(backbone.num_channels, eps=1e-6, data_format="channels_first"),
                nn.Conv1d(backbone.num_channels, self.input_proj_dim, kernel_size=1),
        ) if self.input_proj_dim > 0. else None
        
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.stack_num = backbone.stack_num
        self.frame_skip = backbone.frame_skip

        self.num_txrx = num_txrx
        self.box_feature = box_feature
        
        if box_feature != 'x':
            print("use visual clue")
            vc_input_chan = hidden_dim - self.input_proj_dim #64
            self.vc_input_proj = nn.Sequential(
                nn.Conv2d(256, vc_input_chan, kernel_size=1),
                nn.BatchNorm2d(vc_input_chan, momentum=0.1),
            )

        dummy_input = torch.zeros((128, self.stack_num//self.frame_skip, num_txrx**2, 768))
        
        self.check_dim(dummy_input)

    def forward(self, samples, targets=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        pos, vc_query = None, None
        backbone_out = self.backbone(samples)
        src, mask, pos, vc = backbone_out['src'], backbone_out['mask'], \
                        backbone_out['pos'], backbone_out['pred_feature']             
        assert mask is not None

        src = src.flatten(2)
        if self.input_proj != None:
            input_proj = self.input_proj(src)
            input_proj = rearrange(input_proj, 'b n (t1 t2) -> b n t1 t2', t1=16)

        if vc is not None:
            vc_query = self.vc_input_proj(vc)
            input_proj = torch.cat((input_proj, vc_query), dim=1) if self.input_proj != None else vc_query

        hs = self.transformer(input_proj, mask, self.query_embed.weight, pos)[0]
        hs = hs[-1]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        
        if self.box_feature != 'x':
            out['pred_feature'] = vc

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        
        return out

    def check_dim(self, samples): #samples: NestedTensor):
        print("____check dimension in RPET_____")
        print(f"input shape : {samples.shape}")

        #features= self.backbone(samples)
        pos, vc_query= None, None
        backbone_out = self.backbone(samples)
        src, mask, pos, vc = backbone_out['src'], backbone_out['mask'], \
                        backbone_out['pos'], backbone_out['pred_feature'] 
        print(f"backbone = src : {src.shape}, mask : {mask.shape}")
        if pos is not None:
            print("position embedding = ", pos.shape)
        if vc is not None:
            print("visual clue = ", vc.shape)
            vc_query = self.vc_input_proj(vc)
            print("visual clue to patch = ", vc_query.shape)
        assert mask is not None
        
        src = src.flatten(2)
        if self.input_proj != None:
            input_proj = self.input_proj(src)
            input_proj = rearrange(input_proj, 'b n (t1 t2) -> b n t1 t2', t1=16)
            if vc is not None:
                print(f"input_proj[{input_proj.shape}] + vc_query[{vc_query.shape}]" )
        else:
            print(f"input_proj[X] + vc_query[{vc_query.shape}]" )

        if vc is not None:
            input_proj = torch.cat((input_proj, vc_query), dim=1) if self.input_proj != None else vc_query

        print(f"transofmer input self.input_proj(src) : {input_proj.shape}, mask : {mask.shape}", \
                f"\n self.query_embed.weight {self.query_embed.weight.shape}")
        hs = self.transformer(input_proj, mask, self.query_embed.weight, pos)[0]
        print(f"hs(decoder) = {hs.shape}")
        
        hs = hs[-1] 
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        print("after ffn ", outputs_class.shape, outputs_coord.shape)
        
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, device=None, pose_method=None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        if pose_method == 'hm' or pose_method =='hmdr':
            self.hm_criterion = nn.MSELoss(reduction='mean').to(device)
        if pose_method == 'simdr' or pose_method =='hmdr':
            self.cd_criterion = NMTNORMCritierion(label_smoothing=0.1)

        self.device = device
        if 'feature' in losses:
            self.feature_criterion = nn.MSELoss(reduction='mean').to(device)

        self.pose_method=pose_method

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx][..., :-1], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_pose_dr(self, outputs, targets, indices, num_boxes):
        assert 'x_coord' in outputs

        idx = self._get_src_permutation_idx(indices)
        src_x = outputs['x_coord'][idx]
        src_y = outputs['y_coord'][idx]
        output_size = src_x.shape[2]
        target_pose = torch.cat([t["cd"][J] for t, (_, J) in zip(targets, indices)])
        target_coord = target_pose[:,:,:2]
        target_vis = target_pose[:,:,2]

        target_pose = torch.round(target_coord * output_size)
        target_weight, filtered_joints = filter_target_simdr(target_pose.clone(), target_vis.clone(), output_size)
        target_weight = target_weight.to(src_x)
        filtered_joints = filtered_joints.to(src_x).type(torch.int64)
        loss = self.cd_criterion(src_x, src_y, filtered_joints, target_weight)
        losses = {}
        losses['loss_simdr'] = loss
        return losses

    def loss_pose_hm(self, outputs, targets, indices, num_boxes):
        assert 'pred_hm' in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_pose = outputs['pred_hm'][src_idx]  #(idx_num, 17, 256)
    
        tgt_pose = torch.cat([t["hm"][J] for t, (_, J) in zip(targets, indices)])
        tgt_pose = tgt_pose.to(src_pose) 
        num_joints = src_pose.size(1)
        heatmaps_pred = src_pose.split(1,1)
        heatmaps_gt = tgt_pose.split(1,1) 
        loss = 0
 
        for j in range(num_joints):
            heatmap_pred = heatmaps_pred[j].squeeze()
            heatmap_gt = heatmaps_gt[j].squeeze()
            tmp_loss = 0.5 * self.hm_criterion(heatmap_pred, heatmap_gt)
            loss += tmp_loss

        loss = loss

        losses = {'loss_hm': loss}
        return losses

    def loss_feature(self, outputs, targets, indices, num_boxes):
        assert 'pred_feature' in outputs
    
        src_feature = outputs['pred_feature']
        batch_size = src_feature.size(0)

        tgt_feature = torch.cat([t["features"] for t in targets])
        tgt_feature = tgt_feature.to(src_feature)
        losses = {}
        
        features_pred = src_feature.reshape((batch_size, -1))
        features_gt = tgt_feature.reshape((batch_size, -1))
        feature_loss = self.feature_criterion(features_pred, features_gt)

        losses['loss_feature'] = feature_loss #/ (x*y)
        if indices == None:
            losses['class_error'] = 0.

        return losses

    
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'posedr' : self.loss_pose_dr,
            'posehm' : self.loss_pose_hm,
            'feature': self.loss_feature,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        

        if 'pred_logits' in outputs:
            indices = self.matcher(outputs_without_aux, targets)

            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        else:
            indices = None
            num_boxes = 0
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        
        
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'posehm' or loss =='posedr':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

def soft_nms_pytorch(dets, box_scores, sigma=0.5, thresh=0., cuda=1):
    # Indexes concatenate boxes with the last column
    N = dets.shape[0]
    if cuda:
        #indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1).to(dets)
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    dets = torch.cat((dets, indexes), dim=1)
    #print(dets)
    # The order of boxes coordinate is [y1,x1,y2,x2]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = box_scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        # IoU calculate
        yy1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[pos:, 0].to("cpu").numpy())
        xx1 = np.maximum(dets[i, 1].to("cpu").numpy(), dets[pos:, 1].to("cpu").numpy())
        yy2 = np.minimum(dets[i, 2].to("cpu").numpy(), dets[pos:, 2].to("cpu").numpy())
        xx2 = np.minimum(dets[i, 3].to("cpu").numpy(), dets[pos:, 3].to("cpu").numpy())
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = torch.tensor(w * h).to(dets) if cuda else torch.tensor(w * h)
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    #keep = dets[:, 4][scores > thresh].int()
    keep = dets[:, 4].int()
    

    return dets[:, :4], scores, keep

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes, nms=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        #batch_size = out_bbox.shape[0]
        prob = F.softmax(out_logits, -1)
        #print(prob.shape)
        #print(prob[..., :-1].shape)
        scores, labels = prob[..., :-1].max(-1) # No Obejct는 제외함
        
        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        batch_size, N = boxes.shape[0], boxes.shape[1]
        indices = torch.zeros((batch_size, N))
        if nms:
            for b in range(batch_size):
                dets = boxes[b]
                score = scores[b]
                #print(boxes[b], scores[b])
                boxes[b], scores[b], keep = soft_nms_pytorch(dets, score)
                labels[b] = torch.index_select(labels[b], 0, keep)
                indices[b] = keep
        else:
            for b in range(batch_size):
                indices[b] = torch.arange(0, N, dtype=torch.float)
        #print(indices)
        results = [{'scores': s, 'labels': l, 'boxes': b, 'indices': idx} for s, l, b, idx in zip(scores, labels, boxes, indices)]

        return results



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_rpet(args):
    num_classes = 91    
    device = torch.device(args.device)
    backbone = build_convbackbone(args)

    transformer = build_transformer(args)

    model = RPET(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        box_feature=args.box_feature,
        signal_ratio=args.signal_ratio
    )

    
    if args.pose is not None:
        model = RPETpose(model, freeze_rpet=(args.frozen_weights is not None), 
                            method=args.pose, dr_size=args.dr_size)

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    if args.pose == 'simdr':
        weight_dict["loss_simdr"] = args.pose_loss_coef
    elif args.pose =='hm':
        weight_dict["loss_hm"] = args.pose_loss_coef * 5
    elif args.pose =='hmdr':
        weight_dict["loss_simdr"] = args.pose_loss_coef
        weight_dict["loss_hm"] = args.pose_loss_coef


    if args.box_feature == '16' and args.pose == None:
        weight_dict['loss_feature'] = args.feature_loss_coef


    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    losses = ['labels', 'boxes', 'cardinality']

    if args.pose == 'simdr':
        losses += ['posedr']
    elif args.pose == 'hm':
        losses += ['posehm']
    elif args.pose =='hmdr':
        losses += ['posedr']
        losses += ['posehm']

    if args.box_feature == '16' and args.pose == None:
        losses += ['feature']
        
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                            eos_coef=args.eos_coef, losses=losses, device=device, pose_method=args.pose)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.pose == 'simdr':
        postprocessors['posedr'] = PostProcessPoseDR()
    elif args.pose == 'hm':
        postprocessors['posehm'] = PostProcessPoseHM()
    elif args.pose == 'hmdr':
        postprocessors['posedr'] = PostProcessPoseDR()

        
    print(f"losses = {losses}")
    print(f"weight_dict = {weight_dict}")
    print(f"postprocessors = {postprocessors.keys()}") 

    return model, criterion, postprocessors


