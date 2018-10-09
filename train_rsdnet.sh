#!/bin/bash

# MODIFY CAFFE PATH for YOUR SETTING


CAFFE_ROOT_DIR=PATH_TO_CAFFE/caffe-rsdnet/build/tools/caffe
PATH_TO_SOLVER=./models/solver_rsdnet.prototxt
WEIGHTS=./models/trained_weights/init_resnet.caffemodel


PYTHONPATH=./python_layers/
log_file=rsdnet.log

DEV_ID=0

PYTHONPATH=$PYTHONPATH \
 ${CAFFE_ROOT_DIR} \
 		train -solver ${PATH_TO_SOLVER} \
 			-weights $WEIGHTS \
 					-gpu ${DEV_ID} 2>&1 | tee ${log_file}

