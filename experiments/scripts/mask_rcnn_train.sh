#!/bin/bash
# Usage:
# ./experiments/scripts/mask_rcnn_train.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/mask_rcnn_train.sh cpu 0 VGGnet_train coco

set -x
set -e

export PYTHONUNBUFFERED="True"

DEV=$1
DEV_ID=$2
NET=$3
DATASET=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  coco)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_minival"
    PT_DIR="coco"
    ITERS=50000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/mask_rcnn_train_${NET}_${DEV}.`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./tools/train_net_mask.py --device ${DEV} --device_id ${DEV_ID} \
  --weights data/pretrain_model/VGG16_faster_rcnn.npy \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  --network VGGnet_train \
  ${EXTRA_ARGS}
