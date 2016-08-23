/**
 *
 * This file contains code to compute the adversarial perturbations of
 * the given images. The input can either be a single image or a directory
 * containing one or more images.
 *
 * More information:
 * - The tool is under development and so far it has been tested on
 *   small image batches.
 * - The tool uses the DeepFool algorithm (Moosavi-Dezfooli et al. [1])
 *   and the intension is to provide reusable and fast code that can be
 *   merged with Caffe.
 *
 * [1] http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Moosavi-Dezfooli_DeepFool_A_Simple_CVPR_2016_paper.html
 *
 */

// #include STL headers
#include <string>
#include <vector>
#include <iostream>

// #include support headers
#include <gflags/gflags.h>
#include <glog/logging.h>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif  // USE_OPENCV

// #include caffe files TODO: remove some of them?
#include "caffe/proto/caffe.pb.h"
#include "caffe/net.hpp"
#include "caffe/blob.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/classifier.hpp"
#include "caffe/util/deepfool.hpp"

// using declarations
using caffe::Caffe;
using caffe::Blob;
using caffe::Net;
using caffe::Classifier;
using caffe::DeepFool;
using std::string;
using std::vector;

// FLAGS definitions
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "",
    "The weights of the model to initialize the network.");
DEFINE_string(mean_file, "",
    "The path to the mean image that was used for preprocessing the data.");
DEFINE_string(labels_file, "",
    "The path to the file containing the labels that the network predicts.");
DEFINE_string(images, "",
    "The path to the folder that contains the image(s).");
DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_bool(read_by_folder, true,
    "Optional; it specifies if the data should be read by folder to avoid "
    "memory overflow. Make it false only for small datasets. Default value = "
    "true.");
DEFINE_int32(max_iterations, 50,
    "Optional; the maximum iterations for each image. Default value = 50.");
DEFINE_int32(number_of_labels, 10,
    "Optional; the number of the top classifier that will be used to compute "
    "the input gradient used by the algorithm. Default value = 10.");
DEFINE_int32(batch_size, 128,
    "Optional; the maximum size of the images that will be processed at the "
    "same time during the execution of the algorithm. Is limited by the "
    "availabe memory. Default value = 128.");
DEFINE_int32(Q, 2,
    "Optional; Defualt value = 2. TODO: More info");
DEFINE_double(overshoot, 0.02,
    "Optional; the value of the overshoot that is used in the updating rule "
    "of the DeepFool algorithm. Default value = 0.02.");
DEFINE_double(epsilon, 0.1,
    "Optional; the value of the epsilon which is used to compute the boundary "
    "perturbation of an image. Default value = 0.1");


// function to test the output images
void save_image(cv::Mat img, std::string name, bool FLOAT32_TO_UINT8 = false) {

  cv::Mat tmp_img;
  if (FLOAT32_TO_UINT8) {
    tmp_img.convertTo(img, CV_8UC3);
  } else {
    tmp_img = img;
  }

  cv::imwrite(name, tmp_img);
  std::cout << name << " saved!" << std::endl << std::endl;
}

void DefineMode(const string& gpu_selection) {

  LOG(INFO) << "Classifier constructor";
#ifdef CPU_ONLY
  LOG(INFO) << "Running in CPU.";
  Caffe::set_mode(Caffe::CPU);
#else

  int gpu_id;

  if (gpu_selection.size() == 0) {
    LOG(INFO) << "No GPU selected; try to run on the gpu with id 0.";
    LOG(INFO) << "If you do not want to use the GPU, use the --gpu=no command";
    gpu_id = 0;
  } else if (gpu_selection != "no") {
    gpu_id = boost::lexical_cast<int>(gpu_selection);
  }

  if (gpu_selection == "no") {
    LOG(INFO) << "Running in CPU";
    Caffe::set_mode(Caffe::CPU);
  } else {
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, gpu_id);
    LOG(INFO) << "Running in GPU " << gpu_id << ": " << device_prop.name;

    Caffe::SetDevice(gpu_id);
    Caffe::set_mode(Caffe::GPU);
  }
#endif
}

// function that calls the necessary code
void adversarial_perturbations(DeepFool& deepfool,
                               std::vector<cv::Mat>& input_images,
                               std::vector<std::string>& image_names) {

  deepfool.Preprocess(input_images);
  //deepfool.ImportData(input_images,true);
  Blob<float>* data = caffe::cv_mat_to_blob(input_images);
  deepfool.set_file_names(image_names);
  deepfool.adversarial_perturbations(data);
}

// main tool for computing adversarial images
int main(int argc, char** argv) {

  // Initialization
  ::google::InitGoogleLogging(argv[0]);

  // TODO: Remove the extra information output?
  // FLAG testing
  FLAGS_logtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  // TODO: Make a more concrete message
  gflags::SetUsageMessage("Computes an adversarial data given a dataset and "
      "a network (with its weights)\n"
      "Usage:\n"
      "   compute_adversarial [FLAGS] [NETWORK] [WEIGHTS] [MEAN_FILE]"
      "[LABELS]\n"
      "   the network model needs to contain an image data layer.\n"
      "Optional flags:\n"
      "--max_iterations:\n"
      "--number_of_labels:\n"
      "--batch_size:\n"
      "--Q:\n"
      "--overshoot:\n"
      "--epsilon:\n"
      "   -- The tool is under development at the moment -- ");

  // take the command line flags under the FLAGS_* name and remove them,
  // leaving only the other arguments
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // TODO: REMOVE this - sanity check
  LOG(INFO) << "Just to check the number of the arguments: " << argc;

  // Check whether the number of arguments is the required one
  if (argc > 2) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_adversarial");
    return 1;
  }

  // Check the FLAGS provided to the tool
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition protocol buffer. "
      << "Use the --model flag to specify one.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need a trained model to initialize the"
      << "network. Use the --weights flag to specify a .caffemodel.";
  CHECK_GT(FLAGS_mean_file.size(), 0) << "Need a file containing the mean "
      << "image. Use the --mean_file flag to specify one.";
  CHECK_GT(FLAGS_labels_file.size(), 0) << "Need a file containing the labels.";
  CHECK_GT(FLAGS_images.size(), 0) << "Need a dataset to compute the "
      << "adversarial perturbations";

  DefineMode(FLAGS_gpu);

  // Create and initialize the classifier which will be used
  // to create the adversarial examples
  LOG(INFO) << "Classifier initializing...";

  FLAGS_logtostderr = 0; // disable log to stderr to have a cleaner output

  DeepFool deepfool(FLAGS_model, FLAGS_weights, FLAGS_labels_file,
                    FLAGS_mean_file, FLAGS_number_of_labels, FLAGS_batch_size,
                    FLAGS_max_iterations, FLAGS_Q, FLAGS_overshoot);
  deepfool.set_epsilon(FLAGS_epsilon);

  FLAGS_logtostderr = 1;

  // TODO: Sanity checker
  LOG(INFO) << "Classifier initialized.";

  std::string file_or_dir = FLAGS_images;

  boost::filesystem::path path_to_file_or_dir(FLAGS_images);

  CHECK(boost::filesystem::exists(path_to_file_or_dir))
      << "The path you entered: '" << path_to_file_or_dir << "' does not exist";

  // Three different inputs are supported:
  // (1) a single image file
  // (2) a set of image files contained in a directory
  // (3) a set of image files contained in the sub-directories of a directory
  //     (such as the training set of ILSVRC2012) where it is possible to:
  //         - Compute the adversarial perturbations for one sub-directory
  //           at a time (the recommended method), or
  //         - Compute the adversarial perturbations for all the files in
  //           the sub-directories (but you need to have enough memory)
  if (boost::filesystem::is_regular_file(path_to_file_or_dir)) {

    LOG(INFO) << "The name of the file is " << FLAGS_images;
    vector<cv::Mat> image;
    image.push_back(caffe::read_image(FLAGS_images));
    vector<std::string> image_name;
    image_name.push_back(FLAGS_images);
    adversarial_perturbations(deepfool, image, image_name);

  } else if (boost::filesystem::is_directory(path_to_file_or_dir)) {

    LOG(INFO) << "The image files are contained in " << FLAGS_images;
    LOG(INFO) << "\n\nATTENTION: the tool has been tested only on directories\n"
              << "that contain images or sub-directories that contain\n"
              << "sub-directories with images (one level lower than the\n"
              << "current directory); not on other more general cases such\n"
              << "as directories containing directories of multiple levels.\n";

    typedef boost::filesystem::path bfs_path;
    vector<bfs_path> dir_contents;
    copy(boost::filesystem::directory_iterator(path_to_file_or_dir),
         boost::filesystem::directory_iterator(),
         back_inserter(dir_contents));

    sort(dir_contents.begin(), dir_contents.end());

    // if the directory that is given contains images
    if (boost::filesystem::is_regular_file(dir_contents[0])) {

      LOG(INFO) << "The images are contained in the " << path_to_file_or_dir
                << " directory";

      // TODO: change this to prevend the dual copy()
      vector<cv::Mat> images = caffe::read_images_from_dir(FLAGS_images);
      vector<std::string> image_names =
                                caffe::read_names_from_dir(FLAGS_images);
      adversarial_perturbations(deepfool, images, image_names);

    } else {  // else it is a directory that contains sub-dirs

      LOG(INFO) << "The images are contained in the " << path_to_file_or_dir
                << " sub-directories";

      if (FLAGS_read_by_folder) {
        LOG(INFO) << "Read by folder";
      } else {
        LOG(INFO) << "Use the images of all the subfolders";
      }

      vector<cv::Mat> images;
      vector<std::string> image_names;

      for (vector<bfs_path>::const_iterator it = dir_contents.begin();
           it != dir_contents.end(); ++it) {

        // read one folder at a time and compute the adversarial examples
        // for the images in that folder only
        if (FLAGS_read_by_folder) {
          if (boost::filesystem::is_directory(*it)) {
            DLOG(INFO) << "Adversarial perturbatiosn for " << it->string();
            images = caffe::read_images_from_dir(it->string());
            image_names = caffe::read_names_from_dir(it->string());
            adversarial_perturbations(deepfool, images, image_names);
          }
        } else { // else accumulate all the images and compute for all
          if (boost::filesystem::is_directory(*it)) {
            DLOG(INFO) << "Appending images from " << it->string();
            vector<cv::Mat> tmp_img = caffe::read_images_from_dir(it->string());
            vector<std::string> tmp_name =
                                       caffe::read_names_from_dir(it->string());
            images.insert(images.end(), tmp_img.begin(), tmp_img.end());
            image_names.insert(image_names.end(), tmp_name.begin(),
                               tmp_name.end());
          }
        }
      }

      if (!FLAGS_read_by_folder) {
        adversarial_perturbations(deepfool, images, image_names);
      }
    }
  } else {
    LOG(ERROR) << "The path exists but is neither a regular file nor "
                << "a directory";
  }

  return 0;
}
