# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
import pdb
import cv2

DEBUG = False

def proposal_target_layer(rpn_rois, gt_boxes, gt_masks,_num_classes):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
    # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
    all_rois = rpn_rois
    # TODO(rbg): it's annoying that sometimes I have extra info before
    # and other times after box coordinates -- normalize to one format

    # Include ground-truth boxes in the set of candidate rois
    # gt_boxes: [x1,y1,x2,y2,cls], rpn_rois:[cls,x1,y1,x2,y2] 
    zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
    all_rois = np.vstack(
        (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))  
    )

    # Sanity check: single batch only
    assert np.all(all_rois[:, 0] == 0), \
            'Only single item batches are supported'

    num_images = 1
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Sample rois with classification labels and bounding box regression
    # targets
    #******************************************************
    #  Add gt_masks as input
    #  Also expand labels and masks and output their weights
    #******************************************************
    labels, rois, bbox_targets, bbox_inside_weights, mask_gt, label_weights, mask_weights\
    = _sample_rois(all_rois, gt_boxes, gt_masks, fg_rois_per_image, rois_per_image, _num_classes)

    if DEBUG:
        print 'num fg: {}'.format((labels > 0).sum())
        print 'num bg: {}'.format((labels == 0).sum())
        _count += 1
        _fg_num += (labels > 0).sum()
        _bg_num += (labels == 0).sum()
        print 'num fg avg: {}'.format(_fg_num / _count)
        print 'num bg avg: {}'.format(_bg_num / _count)
        print 'ratio: {:.3f}'.format(float(_fg_num) / float(_bg_num))

    rois = rois.reshape(-1,5)
    # labels = labels.reshape(-1,_num_classes)
    labels = labels.reshape(-1,1)
    bbox_targets = bbox_targets.reshape(-1,_num_classes*4)
    bbox_inside_weights = bbox_inside_weights.reshape(-1,_num_classes*4)
    mask_gt = mask_gt.reshape(-1,cfg.TRAIN.ROI_OUTPUT_SIZE *2,cfg.TRAIN.ROI_OUTPUT_SIZE *2, _num_classes)
    mask_weights = mask_weights.reshape(-1,cfg.TRAIN.ROI_OUTPUT_SIZE *2,cfg.TRAIN.ROI_OUTPUT_SIZE *2, _num_classes)
    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)
    label_weights = label_weights.reshape(-1,_num_classes)

    return rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights,mask_gt,label_weights, mask_weights

def _get_bbox_regression_labels(bbox_target_data, labels_data, mask_gt_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
        labels (ndarray): N x K blob of class labels
        label_weights: N x K blob of label weights
        mask_gt: N x 14 x 14 x K blob of masks
        mask_weights: N x 14 x 14 x K blob of mask weights
    """

    clss = np.array(bbox_target_data[:, 0], dtype=np.uint16, copy=True)
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    mask_gt = np.zeros((clss.size, mask_gt_data.shape[1], mask_gt_data.shape[2], num_classes), dtype=np.float32)
    mask_weights = np.zeros(mask_gt.shape, dtype=np.float32)
    labels = np.zeros((clss.size, num_classes), dtype=np.float32)
    label_weights = np.zeros(labels.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
        mask_gt[ind, :, :, cls] = mask_gt_data[ind, :, :]
        mask_weights[ind, :, :, cls] = np.ones((mask_gt_data.shape[1], mask_gt_data.shape[2]))
        labels[ind, cls] = 1
        label_weights[ind, cls] = 1
    inds = np.where(clss == 0)[0]
    for ind in inds:
	label_weights[ind, 0] = 1

    return bbox_targets, bbox_inside_weights, labels, label_weights, mask_gt, mask_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rois, gt_boxes, gt_masks, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]  # use [0] because max_overlaps is a column vector
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    #*********************
    # sample gt_masks
    # clip to roi region
    # resize to 14*14
    #*********************
    mask_gt_keep = gt_masks[gt_assignment[keep_inds], :, :]
    scale = cfg.TRAIN.ROI_OUTPUT_SIZE*2
    mask_gt_data = np.zeros((len(keep_inds), scale, scale))
    for i in range(len(keep_inds)):
        if labels[i] >0:
            roi = rois[i,1:5]
            if cfg.DEBUG:
                print '_sample_roi'
                print 'i: '+ str(i) +' labels[i]:' + str(labels[i])
                print 'roi' +str(roi[0]) + ' ' +  str(roi[1]) + ' ' + str(roi[2]) + ' ' + str(roi[3])  
            mask_gt_clip = mask_gt_keep[i, int(round(roi[1])) : int(round(roi[3]))+1, int(round(roi[0])) : int(round(roi[2]))+1]
            if cfg.DEBUG:
                print 'mask_gt_keep.shape[1]: ' +str(mask_gt_keep.shape[1])
                print 'mask_gt_keep.shape[2]: ' + str(mask_gt_keep.shape[2])
                print 'mask_gt_clip.shape[0]: ' +str(mask_gt_clip.shape[0])
                print 'mask_gt_clip.shape[1]: ' + str(mask_gt_clip.shape[1])
            fx = float(scale)/mask_gt_clip.shape[1]
            fy = float(scale)/mask_gt_clip.shape[0]
            if cfg.DEBUG:
                print 'mask_gt_clip.shape[0]: ' +str(mask_gt_clip.shape[0])
                print 'mask_gt_clip.shape[1]: ' + str(mask_gt_clip.shape[1])
                print 'scale: ' +str(scale)
                print 'fx:' +str(fx)
                print 'fy:' +str(fy)
            mask_gt_data[i,:,:] = np.round(cv2.resize(mask_gt_clip, None, fx=fx, fy=fy))
        else:
            mask_gt_data[i,:,:] = np.zeros((scale,scale))

    labels_data = labels
    bbox_targets, bbox_inside_weights, labels, label_weights, mask_gt, mask_weights = \
        _get_bbox_regression_labels(bbox_target_data, labels_data, mask_gt_data, num_classes)

    if cfg.TRACE:
        print '========sample rois========'
	print 'fg_inds'
	print fg_inds
	print 'bg_inds'
	print bg_inds
        print 'rois: '
        print rois[0:5,:]
        print 'labels: '
        print labels[0:5,:]
        print 'label_weights: '
        print label_weights[0:5,:]
        print 'bbox_targets: '
        print bbox_targets[0:5,4*59:4*60]
        print 'mask_weighs: '
        print mask_weights[0:5,:,:,59]
        print 'save mask_gt'
        cv2.imwrite('/home/chsiyuan/Documents/542FinalProject/experiments/mask_gt.png',mask_gt[0,:,:,59]*255)


    return labels_data, rois, bbox_targets, bbox_inside_weights, mask_gt, label_weights, mask_weights
