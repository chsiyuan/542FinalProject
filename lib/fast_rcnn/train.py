# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

from fast_rcnn.config import cfg
import gt_data_layer.roidb as gdl_roidb
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
import numpy as np
import os
import tensorflow as tf
import sys
from tensorflow.python.client import timeline
import time

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, sess, saver, network, imdb, roidb, output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        print 'Computing bounding-box regression targets...'
        if cfg.TRAIN.BBOX_REG:  # default is true
            # Calculate targets ??
            # tx = (Gx - Px) / Pw
            # ty = (Gy - Py) / Ph
            # tw = log(Gw / Pw)
            # th = log(Gh / Ph)
            # P should be the output of RPN
            self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        print 'done'

        # For checkpoint
        self.saver = saver

    def snapshot(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
	        # save original values
            with tf.variable_scope('bbox_pred', reuse=True):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")

            orig_0 = weights.eval()
            orig_1 = biases.eval()

            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0 * np.tile(self.bbox_stds, (weights_shape[0], 1))})
            sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1 * self.bbox_stds + self.bbox_means})

            #save ckpt
            filename2 = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                        '_iter_{:d}'.format(iter+1) + '_processed' + '.ckpt')
            filename2 = os.path.join(self.output_dir, filename2)
            self.saver.save(sess, filename2)
            print 'Wrote snapshot to: {:s}'.format(filename2)

            # restore net to original state
	    with tf.variable_scope('bbox_pred', reuse=True):
                sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0})
                sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1})

    def _modified_smooth_l1(self, sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
        """
            ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        """
        sigma2 = sigma * sigma

        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

        smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

        return outside_mul

    # def _binary_mask_loss(self, mask_out, mask_gt, labels, mask_weights):
    #     num_roi = mask_out.get_shape().as_list()[0]
    #     h = mask_out.get_shape().as_list()[1]
    #     w = mask_out.get_shape().as_list()[2]
    #     mask_match = tf.convert_to_tensor(np.zeros((num_roi, h, w)))
    #     mask_weights = tf.convert_to_tensor(np.zeros((num_roi, h, w)))
    #     for i in range(num_roi):
    #         if label[i] > 0:
    #             mask_weights[i,:,:] = tf.convert_to_tensor(np.ones((h, w)))
    #             mask_match[i,:,:] = mask_out[i,:,:,label[i]-1]
    #         else:
    #             mask_match[i,:,:] = mask_out[i,:,:,0]
    #     loss_mask = tf.nn.sigmoid_cross_entropy_with_logits(logits=mask_match, labels=mask_gt)
    #     loss_mask = tf.nn.reduce_mean(tf.multiply(loss_mask, mask_weights))
    #     return loss_mask


    def train_model(self, sess, max_iters):
        """Network training loop."""

        num_classes = self.imdb.num_classes  # 81 classes (read in coco.py)
        #
        # set up RoIDataLayer() class. We will use the function of this class to get next batch of images.
        #
        data_layer = get_data_layer(self.roidb, num_classes)

        if cfg.DEBUG:
            print 'train_model'
            for key in data_layer._roidb[0]:
                print key

        # RPN
        # classification loss
        rpn_cls_score = tf.reshape(self.net.get_output('rpn_cls_score_reshape'),[-1,2])
        rpn_label = tf.reshape(self.net.get_output('rpn-data')[0],[-1])
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_label,-1))),[-1,2])
        rpn_label = tf.reshape(tf.gather(rpn_label,tf.where(tf.not_equal(rpn_label,-1))),[-1])
        rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

        # bounding box regression L1 loss
        rpn_bbox_pred = self.net.get_output('rpn_bbox_pred')
        rpn_bbox_targets = tf.transpose(self.net.get_output('rpn-data')[1],[0,2,3,1])
        rpn_bbox_inside_weights = tf.transpose(self.net.get_output('rpn-data')[2],[0,2,3,1])
        rpn_bbox_outside_weights = tf.transpose(self.net.get_output('rpn-data')[3],[0,2,3,1])

        rpn_smooth_l1 = self._modified_smooth_l1(3.0, rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)
        rpn_loss_box = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, reduction_indices=[1, 2, 3]))

        # R-CNN
        # classification loss (binary sigmoid cross entropy)
        cls_score = self.net.get_output('cls_score')
        # labels = tf.reshape(self.net.get_output('roi-data')[1],[-1, num_classes])
        labels = tf.reshape(self.net.get_output('roi-data')[1],[-1])
        # label_weights = tf.reshape(self.net.get_output('roi-data')[6],[-1, num_classes])
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=labels))

        # cross_entropy_all = tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(logits=cls_score, labels=labels), label_weights)
        # cross_entropy = tf.reduce_mean(tf.reduce_sum(cross_entropy_all, 1))

        # bounding box regression L1 loss
        bbox_pred = self.net.get_output('bbox_pred')
        bbox_targets = self.net.get_output('roi-data')[2]
        bbox_inside_weights = self.net.get_output('roi-data')[3]
        bbox_outside_weights = self.net.get_output('roi-data')[4]

        smooth_l1 = self._modified_smooth_l1(1.0, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
        loss_box = tf.reduce_mean(tf.reduce_sum(smooth_l1, reduction_indices=[1]))

        # mask loss (average binary sigmoid cross entropy)
        mask_out = self.net.get_output('mask_out')
        mask_shape = mask_out.get_shape().as_list()
        mask_gt = tf.reshape(self.net.get_output('roi-data')[5],[-1,mask_shape[1],mask_shape[2],num_classes])
        mask_weights = tf.reshape(self.net.get_output('roi-data')[7],[-1,mask_shape[1],mask_shape[2],num_classes])
        loss_mask_all = tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(logits=mask_out, labels=mask_gt), mask_weights)
        #print mask_out.get_shape().as_list()
        #print mask_gt.get_shape().as_list()
        #print loss_mask_all.get_shape().as_list()
        loss_mask = tf.reduce_mean(tf.reduce_sum(loss_mask_all, 3))

        # l2_loss = cfg.TRAIN.WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        # final loss
        loss = rpn_cross_entropy + rpn_loss_box + cross_entropy + loss_box + loss_mask # + l2_loss

        # Summary
        merged_summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./experiments/summary', sess.graph)

        # optimizer and learning rate
        global_step = tf.Variable(0, trainable=False)
        # lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step,
        #                                cfg.TRAIN.STEPSIZE, cfg.TRAIN.GAMMA, staircase=True)
        # momentum = cfg.TRAIN.MOMENTUM
        # train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(loss, global_step=global_step)
	lr = cfg.TRAIN.LEARNING_RATE
	train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)        

        # intialize variables
        sess.run(tf.global_variables_initializer())
        if self.pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, self.saver, True)

        last_snapshot_iter = -1
        timer = Timer()
        for iter in range(max_iters):
            # get one batch
            # bgr+deformed mask, shape[h,w,4]
            blobs = data_layer.forward()
            # self.snapshot(sess, iter)
            # Make one SGD update
            feed_dict={self.net.data: blobs['data'], self.net.im_info: blobs['im_info'], self.net.keep_prob: 0.5, \
                           self.net.gt_boxes: blobs['gt_boxes'], self.net.gt_masks: blobs['gt_masks']}

            run_options = None
            run_metadata = None
            if cfg.TRAIN.DEBUG_TIMELINE:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            timer.tic()

            rpn_loss_cls_value, rpn_loss_box_value, \
            loss_cls_value, loss_box_value, loss_mask_value, summary, _ \
            = sess.run([rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, loss_mask, merged_summary, train_op],
                        feed_dict=feed_dict,
                        options=run_options,
                        run_metadata=run_metadata)

            # write summary to log file
            train_writer.add_summary(summary, iter)

            if cfg.TRACE:
                print '======calculate loss======'
                cls_score_value, labels_value, \
                label_weights_value, cross_entropy_all_value, _ \
                = sess.run([cls_score, labels, label_weights, cross_entropy_all, train_op],
                            feed_dict=feed_dict,
                            options=run_options,
                            run_metadata=run_metadata)
                print 'cls_score[59]: '
                print cls_score_value[:,59]
		print 'cls_score[0]: '
		print cls_score_value[:,0]
                print 'labels[59]: '
                print labels_value[:,59]
		print 'labels[0]: '
		print labels_value[:,0]
                print 'label_weights_value[59]: '
                print label_weights_value[:,59]
		print 'label_weights_value[0]: '
		print label_weights_value[:,0]
                print 'cross_entropy_all_value[59]: '
                print cross_entropy_all_value[:,59]
		print 'cross_entropy_all_value[0]: '
		print cross_entropy_all_value[:,0]

            # mask_gt_value, mask_weights_value, _ \
            # = sess.run([mask_gt, mask_weights, train_op],
            #             feed_dict=feed_dict,
            #             options=run_options,
            #             run_metadata=run_metadata)

            timer.toc()

            if cfg.TRAIN.DEBUG_TIMELINE:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open(str(long(time.time() * 1000)) + '-train-timeline.ctf.json', 'w')
                trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                trace_file.close()

            if (iter+1) % (cfg.TRAIN.DISPLAY) == 0:
                print 'iter: %d / %d, total loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f, loss_mask: %.4f, lr: %f'%\
                (iter+1, max_iters, rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value + loss_mask_value, \
                    rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, loss_mask_value, lr)

                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    # HAS_RPN = True when training
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            gdl_roidb.prepare_roidb(imdb)
        else:
            rdl_roidb.prepare_roidb(imdb)
    else:
        rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb


def get_data_layer(roidb, num_classes):
    """return a data layer."""
    # HAS_RPN = True when training
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            layer = GtDataLayer(roidb)
        else:
            layer = RoIDataLayer(roidb, num_classes)
    else:
        # layer._roidb = roidb
        # layer._num_classes = num_classes
        # layer._shuffle_roidb_inds()
        layer = RoIDataLayer(roidb, num_classes)

    return layer

def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI

        # For VOC, overlaps are all = 1
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb


def train_net(network, imdb, roidb, output_dir, pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""
    if cfg.DEBUG:
        print 'train_net before filter_roidb'
        for key in roidb[0]:
            print key
    roidb = filter_roidb(roidb)
    if cfg.DEBUG:
        print 'train_net after filter_roidb'
        for key in roidb[0]:
            print key
    saver = tf.train.Saver(max_to_keep=100)  # the maximum number of check point files is 100
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sw = SolverWrapper(sess, saver, network, imdb, roidb, output_dir, pretrained_model=pretrained_model)
        print 'Solving...'
        sw.train_model(sess, max_iters)
        print 'done solving'
