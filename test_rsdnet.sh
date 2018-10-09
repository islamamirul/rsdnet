#!/bin/bash

# MODIFY PATH for YOUR SETTING
 
PATH_TO_SCRIPT=./scripts/inference/test_rsdnet.py
PATH_TO_MODEL=./models/test_rsdnet.prototxt
PATH_TO_WEIGHTS=./models/trained_weights/rsdnet.caffemodel

PYTHONPATH=./python_layers/

PYTHONPATH=$PYTHONPATH python ${PATH_TO_SCRIPT} --model ${PATH_TO_MODEL} --weights ${PATH_TO_WEIGHTS} --iter 425 

