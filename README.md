# deepfool_caffe

# deepfool_caffe

This repository contains a C++ implementation of the [DeepFool algorithm](http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Moosavi-Dezfooli_DeepFool_A_Simple_CVPR_2016_paper.html) by Moosavi-Dezfooli, Fawzi, Frossard. The implementation is based on the [Caffe](https://github.com/BVLC/caffe) deep learning framework and it provides a MATLAB API.

To use the algorithm you need to:
 1. Install Caffe
 2. Download/clone the project
 3. Open the `import_deepfool_caffe.sh` file and set the path to your Caffe implementation at the CAFFE_PATH variable
 4. Run: `./import_deepfool_caffe.sh` (ATTENTION: Running the script will copy the necessary files in the Caffe folder and it will also *update/change* the `$CAFFE_PATH/matlab/+caffe/private/caffe_.cpp` file. None of the previous functionality is removed, so it should be fine.)

Then you can use the MATLAB interface, where you can:
```
% Define a DeepFool object:
df = caffe.DeepFool(model_file, net_file, labels_file);

% Use the object to get the perturbations of a single or more images that are in a 4-D array
% a is a 4-D MATLAB array with the images as the Caffe requires it
% (more one http://caffe.berkeleyvision.org/tutorial/interfaces.html )
perturbations = df.adversarial_perturbations(a);

% Use the object to get the perturbations, the number of iterations, the fooling label
% and the true label of the images
[pert, iter, foollab, truelab] = df.adversarial_perturbations(a);
```

You can also see the `deepfool_example.m` as an example of how to use the tool.

*The code is under development. Also it not properly cleaned and documented. Additional documentation and fixes are comming soon.*
