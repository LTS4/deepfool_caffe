#!/usr/bin/env sh
# This file is used to copy the files from the src/ directory
# to the necessary caffe directory; please specify the current
# caffe directory

CAFFE_PATH=/path/to/caffe
cp src/caffe_.cpp $CAFFE_PATH/matlab/+caffe/private
cp src/classifier.cpp $CAFFE_PATH/src/caffe/util
cp src/classifier.hpp $CAFFE_PATH/include/caffe/util
cp src/compute_adversarial.cpp $CAFFE_PATH/tools
cp src/deepfool.cpp $CAFFE_PATH/src/caffe/util
cp src/deepfool.hpp $CAFFE_PATH/include/caffe/util
cp src/DeepFool.m $CAFFE_PATH/matlab/+caffe
cp src/get_deepfool.m $CAFFE_PATH/matlab/+caffe

make -C $CAFFE_PATH
make matcaffe -C $CAFFE_PATH
