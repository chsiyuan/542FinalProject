import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
from networks.factory import get_network


CLASSES = ('__background__',
           'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
           'stop sign', 'parking meter',  'bench',  'bird',  'cat',  
           'dog',  'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
           'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
           'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
           'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 
           'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 
           'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
           'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
           'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
           'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
           'hair drier', 'toothbrush')


#CLASSES = ('__background__','person','bike','motorbike','car','bus')

def vis_detections(im, class_name, dets, segs, seg_together, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        seg = segs[i, :, :]

        # Recover the mask to the original size of bbox
        seg_h = seg.shape[1]
        seg_w = seg.shape[0]
        height = bbox[2]-bbox[0]
        width  = bbox[3]-bbox[1]
        fx = width/seg_w
        fy = height/seg_h
        seg_resize = cv2.resize(seg, None, fx=fx, fy=fy)

        # Map the resized mask to the original image


        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def test_single_frame(sess, net, image_name, mask, force_cpu):
    """Detect object classes in an image using pre-computed object proposals."""

    #********************
    # Need change here
    #********************
    im_file = os.path.join(cfg.DATA_DIR, image_name)
    #im_file = os.path.join('/home/corgi/Lab/label/pos_frame/ACCV/training/000001/',image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes, masks = im_detect(sess, net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    seg_together = np.zeros((im.shape[0],im.shape[1]))
    fig, ax = plt.subplots(figsize=(12, 12))
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        cls_masks = masks[:, :, :, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH, force_cpu)
        dets = dets[keep, :]
        segs = cls_masks[keep, :, :]
        print ('After nms, {:d} object proposals').format(dets.shape[0])
        seg_together = vis_detections(im, cls, dets, segs, seg_together, ax, thresh=CONF_THRESH)

    return next_mask

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default=' ')

    args = parser.parse_args()

    return args
if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    force_cpu = False
    if args.cpu_mode == True:
        print '\nUse cpu mode\n'
        cfg.USE_GPU_NMS = False
        force_cpu = True
    else:
        print '\nUse gpu mode\n'
        cfg.USE_GPU_NMS = True
        cfg.GPU_ID = args.gpu_id

    if args.model == ' ':
        raise IOError(('Error: Model not found.\n'))

    output_dir = '../results/test'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    input_dir = '../data/'
    if not os.path.exists(input_dir):
        raise IOError(('Error: Input not found.\n'))
        
    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network(args.demo_net)
    # load model
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    saver.restore(sess, args.model)
   
    #sess.run(tf.initialize_all_variables())

    print '\n\nLoaded network {:s}'.format(args.model)

    # Warmup on a dummy image
    # im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    # for i in xrange(2):
    #     _, _= im_detect(sess, net, im)

    images = os.listdir(input_dir) 
    # load mask of first frame
    mask = cv2.imread('',0)
    for image in images: 
        if not os.path.isdir(file):
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Segmentation for {}'.format(im_name)
        next_mask = test_single_frame(sess, net, im_name, mask, force_cpu)
        # To be implementeds
        mask = mask_deformation(next_mask)
