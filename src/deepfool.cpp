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

#include <string>
#include <numeric> // inner_product
#include <cmath>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif

#include <boost/lexical_cast.hpp>

#include <caffe/util/math_functions.hpp>
#include <caffe/net.hpp>
#include <caffe/util/deepfool.hpp>
#include <caffe/util/classifier.hpp>

namespace caffe{

DeepFool::DeepFool(const std::string& model_file,
                   const std::string& trained_file,
                   const std::string& label_file,
                   const std::string& mean_file,
                   const size_t number_of_labels,
                   const size_t batch_size,
                   const int MAX_ITER,
                   const int Q,
                   const float OVERSHOOT)
    : classifier_(model_file, trained_file, label_file, mean_file) {

  number_of_labels_ = number_of_labels;
  batch_size_ = batch_size;
  max_iterations_ = MAX_ITER;
  Q_ = Q;
  overshoot_ = OVERSHOOT;
  epsilon_ = 0.5;
}

// TODO: REMOVE this function; used only for verification
void print_vector(const vector<float> vec, size_t init = 0, size_t sz = 30) {

  for (size_t i = init; i != sz; ++i) {
    if (i % 10 == 0)
      std::cout << std::endl;
    std::cout << vec[i] << "  ";
  }

  std::cout << std::endl << std::endl;

}

// TODO: Maybe change the name?
//       Maybe change the functionality?
//       Maybe decouple the second functionality?
// It works as it is, but it might need some changes in the future
size_t Argmin(const vector<float>& f_p, const vector<float>& w_pn,
              bool RETURN_CLASSIFIER_INDEX=false, int top_classifier_index=-1) {

  CHECK(f_p.size() == w_pn.size()) << "(ArgMin) different vector dimensions";

  vector<float> tmp;
  for (size_t i = 0; i < f_p.size(); ++i) {
    tmp.push_back(std::abs(f_p[i]) / w_pn[i]);
  }

  // std::cout << "Argmin" << std::endl;
  // print_vector(tmp, 0, f_p.size());

  vector<float>::const_iterator min_value =
                                        min_element(tmp.begin(), tmp.end());

  CHECK(min_value != tmp.end()) << "(ArgMin) no minimum value; should not occur";

  size_t idx = min_value - tmp.begin();

  // Return the classifier index instead of the
  // minimum index in the vectors.
  if (RETURN_CLASSIFIER_INDEX) {
    CHECK(top_classifier_index > 0) << "Please provide the top classifier";
    if (idx >= top_classifier_index) {
      idx++;
    }
  }

  return idx;

}

// TODO: Find a better way: use transformation() or BLAS (or caffe's BLAS)
vector<float> subtract_vec(const vector<float>& v1,
                           const vector<float>& v2) {

  CHECK(v1.size() == v2.size())
      << "(DeepFool) Vectors are not of the same size";

  vector<float> result;

  for (size_t i = 0; i < v1.size(); ++i) {
    result.push_back(v1[i] - v2[i]);
  }

  return result;
}

// TODO: maybe use BLAS for computation?
float vector_norm(const vector<float>& v) {

  return std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
  // float sum = 0.0;
  //
  // for (size_t i = 0; i != v.size(); ++i) {
  //   sum += pow(v[i], 2);
  // }
  //
  // return sqrt(sum);
}

// TODO: REMOVE this function
void print_img_dim(const cv::Mat& image) {

  std::cout << image.dims << " dimensions and " << std::endl
            << image.size() << std::endl << " and "
            << image.channels() << " channels." << std::endl;

}

void DeepFool::reshapeGradient(const vector<float>& w_prime,
                               caffe::Blob<float>* grd) {

  vector<int> sh = grd->shape();
  float* data = grd->mutable_cpu_data();

  size_t N = sh[0];
  size_t K = sh[1];
  size_t H = sh[2];
  size_t W = sh[3];

  // n = 0 always; it is just written for convinience
  for (size_t n = 0; n < sh[0]; ++n) {
    for (size_t k = 0; k < sh[1]; ++k) {
      for (size_t h = 0; h < sh[2]; ++h) {
        for (size_t w = 0; w < sh[3]; ++w) {
          // data[n*K*W*H + k*H*W + h*W + w]
          //   = (1 + overshoot_)*w_prime[n*K*W*H + w*K + h*W*K + k];
          data[n*K*W*H + k*H*W + h*W + w]
          = (1 + overshoot_)*w_prime[n*K*W*H + k*W*H + h*W + w];
        }
      }
    }
  }
}

void DeepFool::RemoveImages(const vector<int>& finished_idx,
                            Blob<float>* data, vector<int>& image_idx,
                            vector<Classifier::Prediction>& top_label_prob,
                            vector<int >& top_classifier_index,
                            vector<vector<int> >& top_N_classifiers_index,
                            size_t image_size) {

  Blob<float>* res = new Blob<float>(data->shape(0) - finished_idx.size(),
                                     data->shape(1), data->shape(2),
                                     data->shape(3));


  DLOG(INFO) << finished_idx.size() << " will be removed";
  DLOG(INFO) << "The size of the new blob " << res->shape_string();
  vector<int>::iterator it_image_idx = image_idx.begin();
  vector<int>::iterator it_top_classifier_index = top_classifier_index.begin();
  vector<Classifier::Prediction>::iterator it_top_label_prob = top_label_prob.begin();
  vector<vector<int> >::iterator it_top_N_classifiers_index_row = top_N_classifiers_index.begin();
  size_t f_idx = 0;
  size_t j = 0;
  for (size_t i = 0; i < data->shape(0); ++i) {
    if (i != finished_idx[f_idx]) { // problematic -> check before call
      caffe_copy(image_size, data->cpu_data() + i*image_size,
                             res->mutable_cpu_data() + j*image_size);
      ++j;
      ++it_image_idx;
      ++it_top_label_prob;
      ++it_top_classifier_index;
      ++it_top_N_classifiers_index_row;
    } else {
      it_image_idx = image_idx.erase(it_image_idx);
      it_top_label_prob = top_label_prob.erase(it_top_label_prob);
      it_top_classifier_index = top_classifier_index.erase(it_top_classifier_index);

      for (size_t k = 0; k < top_N_classifiers_index.size(); ++k) {
        vector<int>::iterator it = top_N_classifiers_index[k].begin();
        top_N_classifiers_index[k].erase( top_N_classifiers_index[k].begin() + j);
      }

      LOG(INFO) << "Removing " << i << " == " << finished_idx[f_idx] << " from the data";
      f_idx++;
    }
  }
  LOG(INFO) << "Finished";

  data->ReshapeLike(*res);
  if (res->count() != 0) {
    caffe_copy(res->count(), res->cpu_data(), data->mutable_cpu_data());
  }
  delete res;
}

void DeepFool::AddImages(size_t data_to_add, Blob<float>* x, vector<int>& image_idx,
                         vector<Classifier::Prediction>& top_label_prob,
                         vector<int>& top_classifier_index,
                         vector<vector<int> >& top_N_classifiers_index,
                         size_t image_size, Blob<float>* data,
                         size_t current_data_read,
                         size_t& current_index) {

  size_t x_data_num = x->shape(0);
  x->Reshape(data_to_add+x_data_num, data->shape(1), data->shape(2), data->shape(3));
  DLOG(INFO) << "x shape will be: " << x->shape_string();
  DLOG(INFO) << "current_data_read: " << current_data_read << " x_data_num "
            << x_data_num << " data_to_add:" << data_to_add;
  caffe_copy(data_to_add*image_size, data->cpu_data() + current_data_read * image_size,
                                     x->mutable_cpu_data() + x_data_num*image_size);

   Blob<float>* tmp_data = new Blob<float>(data_to_add, data->shape(1),
                                           data->shape(2), data->shape(3));

   caffe_copy(data_to_add*image_size, data->cpu_data() + current_data_read * image_size,
                                      tmp_data->mutable_cpu_data());
   classifier_.ImportData(tmp_data);
   vector<vector<Classifier::Prediction> > tmp_pred = classifier_.Classify(number_of_labels_);

   for (size_t i = 0; i < data_to_add; ++i) {
      top_label_prob.push_back(tmp_pred[i][0]);
      top_classifier_index.push_back(
                              classifier_.get_label_index(top_label_prob[i].first));
      image_idx.push_back(current_index++);
      for (size_t j = 1; j < number_of_labels_; ++j) {
        top_N_classifiers_index[j-1].push_back(classifier_.get_label_index(tmp_pred[i][j].first));
      }
   }
}


void DeepFool::ImportData(const std::vector<cv::Mat> data, bool RESHAPE) {
  classifier_.ImportData(data, RESHAPE);
}

Blob<float>* cv_mat_to_blob(std::vector<cv::Mat> data) {

  Blob<float>* img_blob = new Blob<float>(data.size(), data[0].channels(),
                                          data[0].rows, data[0].cols);

  float* img_blob_data = img_blob->mutable_cpu_data();

  int N = data.size();
  int K = data[0].channels();
  int H = data[0].rows;
  int W = data[0].cols;

  for (size_t n = 0; n < N; ++n) {

    CHECK(data[n].type() == CV_32FC3)
        << "Image data should be in CV_32FC3 format to import it";

    for (size_t h = 0; h < H; ++h) {
      const float* img_row = data[n].ptr<float>(h);

      for (size_t w = 0; w < W; ++w) {
        for (size_t k = 0; k < K; ++k) {
          // same as:
          // input_data[((n*K + k)*H + h)*W + w] = img_row[w*K + k];
          img_blob_data[n*K*W*H + k*H*W + h*W +w] = img_row[w*K + k];
        }
      }
    }
  }
  return img_blob;
}


std::string
DeepFool::print_predictions(vector<vector<Classifier::Prediction> > predictions,
                            vector<int> image_idx, size_t topN) {

  LOG(INFO) << "Printing predictions for the " << topN
            << " predictors of each data point (image).";

  LOG(INFO) << "Predictions " << predictions.size()
            << " Image_idx " << image_idx.size()
            << " image_idx[0] " << image_idx[0]
            << " image_idx[1] " << image_idx[1]
            << " topN " << topN;
  size_t number_of_data = predictions.size();
  std::string output("\n");

  topN = std::min(topN, predictions[0].size());

  if (file_names_.size() == 0) {
    for (size_t i = 0; i < number_of_data; ++i) {
      std::ostringstream os, tmp;
      tmp << " - " << i << " image has ";

      for (size_t j = 0; j < topN; ++j) {
        os << j << " prediction is "
          << predictions[i][j].second << " for the label "
          << predictions[i][j].first << "\n";
          if (j != topN-1) {
            os << string(tmp.str().size(), ' ');
          }
        }
        output +=  tmp.str() + os.str();
    }
  } else {

    for (size_t i = 0; i < number_of_data; ++i) {
      std::ostringstream os, tmp;
      tmp << " - " << i << "  " << file_names_[image_idx[i]] << " image has ";

      for (size_t j = 0; j < topN; ++j) {
        os << (j + 1) << " prediction is "
          << predictions[i][j].second << " for the label "
          << predictions[i][j].first << "\n";
          if (j != topN-1) {
            os << string(tmp.str().size(), ' ');
          }
        }
        output +=  tmp.str() + os.str();
    }
  }

  LOG(INFO) << "Output size: " << output.size();

  return output;
}

/****************************************************************************/

Blob<float>*
DeepFool::adversarial_perturbations(Blob<float>* input_data) {
    vector<int> iterations(input_data->shape(0));
    vector<int> fooling_label(input_data->shape(0));
    vector<int> correct_label(input_data->shape(0));
    return adversarial_perturbations(iterations, fooling_label,
                                     correct_label, input_data);
}

Blob<float>*
DeepFool::adversarial_perturbations(std::vector<int>& iterations,
                                    std::vector<int>& fooling_label,
                                    std::vector<int>& correct_label,
                                    Blob<float>* input_data) {

  Blob<float>* data = new Blob<float>(0,0,0,0);

  LOG(INFO) << "Computing adversarial examples using the DeepFool algorithm "
            << "by computing the gradients of the top " << number_of_labels_
            << " classifiers for image";

  // TODO: Test this part
  // if no data is provided then load whatever is in the input blob
  if (input_data == NULL) {

    Blob<float>* net_input_blob = classifier_.get_input_blob();

    // TODO: Better condition?
    // Check if there is anything in the input blob
    // if (input_blob->cpu_data() == NULL) {
    //   LOG(ERROR) << "The first layer does not contain any data and no "
    //              << "data has been provided as input.";
    // }

    data->Reshape(net_input_blob->shape());
    caffe_copy(net_input_blob->count(), net_input_blob->cpu_data(),
                                        data->mutable_cpu_data());
  } else { // else copy the provided data
    data->Reshape(input_data->shape());
    caffe_copy(input_data->count(), input_data->cpu_data(),
                                    data->mutable_cpu_data());
  }

  // Information of the data to be processed; their number and
  // the size of each data point (image dimensions most likely)
  size_t number_of_data = data->shape(0);
  size_t image_size = data->shape(1)*data->shape(2)*data->shape(3);

  LOG(INFO) << "Number of data to be processed: " << number_of_data
            << " with each data point (images) having size: " << image_size;

  vector<string> labels = classifier_.get_labels(); // TODO: Move somewhere else>

  // import number_of_data or batch_size_ data to
  // the blob that is used for computations
  size_t max_num_data = std::min(batch_size_, number_of_data);

  DLOG(INFO) << max_num_data << " will be processed at the same time";

  // create the blob that will temporarily hold the images
  // that will be processed simultaneously
  Blob<float>* x = new Blob<float>(0,0,0,0);
  x->Reshape(max_num_data, data->shape(1), data->shape(2), data->shape(3));

  caffe_copy(x->count(), data->cpu_data(), x->mutable_cpu_data());

  DLOG(INFO) << "Created the x blob (image vector) with shape "
             << x->shape_string() << " which contains the first "
             << max_num_data << " of the (input) data blob.";

  classifier_.ImportData(x, true);

  vector<vector<Classifier::Prediction> >
                            tmp_pred =  classifier_.Classify(number_of_labels_);

  // initialize the vectors which will hold informations about the
  // images that are being processed;
  vector<int> image_idx;
  vector<int> image_iter;
  size_t current_index = 0;
  vector<Classifier::Prediction> top_label_prob;
  vector<int> top_classifier_index;
  vector<vector<int> > top_N_classifiers_index(number_of_labels_-1);

  for (size_t i = 0; i < max_num_data; ++i) {
    image_idx.push_back(current_index++);
    image_iter.push_back(0);
    top_label_prob.push_back(tmp_pred[i][0]);
    top_classifier_index.push_back(
                      classifier_.get_label_index(top_label_prob[i].first));
    for (size_t j = 1; j < number_of_labels_; ++j) {
      top_N_classifiers_index[j-1].push_back(classifier_.get_label_index(tmp_pred[i][j].first));
    }
  }

  for (size_t i = 0; i < iterations.size(); ++i) {
    iterations[i] = 0;
  }

  LOG(INFO) << print_predictions(tmp_pred, image_idx, 1);

  // Blob where the final perturbation for each image is accumulated
  Blob<float>* r_final = new Blob<float>(data->shape(0), data->shape(1),
                                         data->shape(2), data->shape(3));
  caffe_set(r_final->count(), (float) 0.0, r_final->mutable_cpu_data());

  // TODO: REMOVE
  vector<vector<float> > r(number_of_data,
                           vector<float>(image_size, float()));

  vector<size_t> image_iterations(number_of_data);

  size_t images_processed = 0;
  while (images_processed != number_of_data) {

    // concat with next command
    vector<vector<float> > current_top_classifier_grad;

    // Compute the gradient w.r.t the input of each top classifier
    DLOG(INFO) << "Computing adversarial for: " << x->shape(0) << " images";
    classifier_.ImportData(x,true);

    current_top_classifier_grad =
                  classifier_.InputGradientofClassifier2(top_classifier_index);

    // // TODO: REMOVE; only for verification
    // for (size_t i = 0; i < x->shape(0); ++i) {
    //   std::cout << "Printing current_top_classifier_grad for " << i
    //             << " with top classifier " << top_classifier_index[i]
    //             << std::endl;
    //   print_vector(current_top_classifier_grad[i], 0, 300);
    // }

    // TODO: For verification; REMOVE
    DLOG(INFO) << "current_top_classifier_grad has size: " << current_top_classifier_grad.size();

    for (size_t i = 0; i != current_top_classifier_grad.size(); ++i) {
      DLOG(INFO) << "current_top_classifier_grad[" << i << "] has size " << current_top_classifier_grad[i].size();
    }

    vector<vector<float> > f_prime(x->shape(0));//, vector<float>(labels.size()-1, float()));
    vector<vector<float> > w_prime_norm(x->shape(0));//, vector<float>(labels.size()-1, float()));

    vector<vector<float> > predictions = classifier_.Predict();

    for (size_t k = 0; k < top_N_classifiers_index.size(); ++k) {

      if (number_of_labels_ > 100) {
        LOG_EVERY_N(INFO, 10) << "Computing for the " << k << "-th classifier"
                 << " (and the next few of them) of the total of "
                 << top_N_classifiers_index.size() << " iterations";
      } else {
        LOG(INFO) << "Computing for the " << k << "-th classifier of "
                  << top_N_classifiers_index.size() << " iterations";
      }

      vector<vector<float> > k_classifier_grad =
            classifier_.InputGradientofClassifier2(top_N_classifiers_index[k]);

      // TODO: Better messages
      // CHECK(k_classifier_grad.size() == x.size()) << "ERROR 1";
      // CHECK(k_classifier_grad[0].size() ==
      //       current_top_classifier_grad[0].size()) << "ERROR 2";

      // for each image in the batch compute the f' and the
      // (normalized) w' needed to compute the minimum perturbation
      for (size_t i = 0; i < x->shape(0); ++i) {
        //if (labels[k] != top_label_prob[i].first) {
          f_prime[i].push_back(predictions[i][top_N_classifiers_index[k][i]] - top_label_prob[i].second);

          w_prime_norm[i].push_back(vector_norm(subtract_vec(k_classifier_grad[i],
                                            current_top_classifier_grad[i])));
        // } else {
        //   std::cout << "Not computing for image " << i
        //             << " with label " << top_label_prob[i].first
        //             << std::endl;
        // }
      }
    }

    // TODO: For verification
    DLOG(INFO) << "f_prime has size: " << f_prime.size();
    DLOG(INFO) << "f_prime[0] has size " << f_prime[0].size();

    DLOG(INFO) << "w_prime_norm has size: " << w_prime_norm.size();
    DLOG(INFO) << "w_prime_norm[0] has size " << w_prime_norm[0].size();

    vector<int> l_hat;
    vector<float> tmp_mul;
    vector<vector<float> > w_prime;
    vector<int> argmin_classifier_index;
    // for each image in the batch
    for (size_t i = 0; i != x->shape(0); ++i) {

      DLOG(INFO) << top_N_classifiers_index[0][i] << " "
                  << top_N_classifiers_index[1][i] << " "
                  << top_N_classifiers_index[2][i] << " "
                  << top_N_classifiers_index[3][i] << " ";

      // std::cout << "Printing f_prime for " << i << std::endl;
      // print_vector(f_prime[i], 0, f_prime[i].size());
      //
      // std::cout << "Printing w_prime_norm for " << i << std::endl;
      // print_vector(w_prime_norm[i], 0, w_prime_norm[i].size());

      float min_clas = Argmin(f_prime[i], w_prime_norm[i]);
      DLOG(INFO) << "min_clas " << min_clas;
      l_hat.push_back(min_clas);

      argmin_classifier_index.push_back(top_N_classifiers_index[min_clas][i]);

      DLOG(INFO) << "The index of the minimum classifier is "
                << top_N_classifiers_index[min_clas][i] <<  " of the image " << i;
      DLOG(INFO) << "The values for " << l_hat[i]
                << " are:  f_prime: " << f_prime[i][l_hat[i]]
                << " and w_prime: " << w_prime_norm[i][l_hat[i]];
      tmp_mul.push_back(
            std::abs(f_prime[i][l_hat[i]]) / std::pow(w_prime_norm[i][l_hat[i]],2));
    }

      // Blob<float>* x_i = new Blob<float>(0,0,0,0);
      // x_i->Reshape(1, x->shape(1), x->shape(2), x->shape(3));
      // caffe_copy(image_size, x->cpu_data()+i*image_size, x_i->mutable_cpu_data());
      // classifier_.ImportData(x_i,true);

  vector<vector<float> > argmin_grads =
              classifier_.InputGradientofClassifier2(argmin_classifier_index);

  Blob<float>* tmp_x = new Blob<float>(x->shape(0), x->shape(1),
                                       x->shape(2), x->shape(3));

  for (size_t i = 0; i != x->shape(0); ++i) {
      w_prime.push_back(
            subtract_vec(argmin_grads[i], current_top_classifier_grad[i]));

      DLOG(INFO) << "l_hat size " << l_hat.size();
      DLOG(INFO) << "tmp size " << tmp_mul.size();
      DLOG(INFO) << "tmp value " << tmp_mul[i];
      DLOG(INFO) << "w_prime size " << w_prime.size();
      DLOG(INFO) << "w_prime[i] size: " << w_prime[i].size();
      DLOG(INFO) << "r size " << r.size();
      DLOG(INFO) << "r[i] size " << r[i].size();

      for (size_t j = 0; j < w_prime[i].size(); ++j) {
        w_prime[i][j] = w_prime[i][j] * tmp_mul[i];
        r[image_idx[i]][j] += w_prime[i][j];
      }
      //print_vector(w_prime[i], 200, 700);

      // cv::Mat r_image(classifier_.get_geometry().height,
      //                 classifier_.get_geometry().width,
      //                 CV_32FC3, w_prime[i].data());
      //
      // std::cout << "r_image at iteration " << i << " has " << std::endl;
      // print_img_dim(r_image);
      //
      // std::cout << "x[" << i << "] at iteration " << general_iter << " has " << std::endl;
      // print_img_dim(x[i]);

      Blob<float>* grd = new Blob<float>(0,0,0,0);
      grd->Reshape(1, x->shape(1), x->shape(2), x->shape(3));
      reshapeGradient(w_prime[i], grd);
      DLOG(INFO) << "grd size: " << grd->shape_string();

      caffe_add(image_size, x->cpu_data() + i*image_size, grd->cpu_data(),
                            tmp_x->mutable_cpu_data() + i*image_size);
      delete grd;

      // float ww[w_prime[i].size()];
      // std::copy(w_prime[i].begin(), w_prime[i].end(), ww);
      float tmp_overshoot = overshoot_;
      overshoot_ = 0;
      grd = new Blob<float>(0,0,0,0);
      grd->Reshape(1, x->shape(1), x->shape(2), x->shape(3));
      reshapeGradient(w_prime[i], grd);
      DLOG(INFO) << "grd size: " << grd->shape_string();
      caffe_add(image_size, x->cpu_data() + i*image_size, grd->cpu_data(),
                            x->mutable_cpu_data() + i*image_size);
      delete grd;

      // TODO: More efficient
      grd = new Blob<float>(0,0,0,0);
      grd->Reshape(1, x->shape(1), x->shape(2), x->shape(3));
      reshapeGradient(w_prime[i], grd);
      caffe_add(image_size, r_final->cpu_data() + image_idx[i]*image_size,
        grd->cpu_data(), r_final->mutable_cpu_data() + image_idx[i]*image_size);
      delete grd;
      overshoot_ = tmp_overshoot;

      // std::string name = "Image_" +
      //                     boost::lexical_cast<std::string>(i) +
      //                     "_iteration_" +
      //                     boost::lexical_cast<std::string>(general_iter) +
      //                     ".jpg";
      // cv::imwrite(name, x[i] + classifier_.get_mean());
      //
      // std::string name2 = "Perturbation_for_image_" +
      //                     boost::lexical_cast<std::string>(i) +
      //                     "_iteration_" +
      //                     boost::lexical_cast<std::string>(general_iter) +
      //                     ".jpg";
      // cv::imwrite(name2, r_image*256);
    }

    size_t itx = 0;
    vector<int> finished_idx;
    // Erase the images whose adversarial perturbations
    // have been found and save the perturbations
    classifier_.ImportData(tmp_x, true);
    vector<vector<Classifier::Prediction> > test_predictions =
                                                      classifier_.Classify(1);
    classifier_.ImportData(x,true);
    vector<vector<Classifier::Prediction> > tmp_predictions =
                                                      classifier_.Classify(3);

    DLOG(INFO) << print_predictions(tmp_predictions, image_idx, 3);
    while (itx < x->shape(0)) {
      if (test_predictions[itx][0].first != top_label_prob[itx].first) {
        finished_idx.push_back(itx);
        if (tmp_predictions[itx][0].first != top_label_prob[itx].first) {
          fooling_label[image_idx[itx]] =
                classifier_.get_label_index(tmp_predictions[itx][0].first);
        } else {
          fooling_label[image_idx[itx]] =
                classifier_.get_label_index(tmp_predictions[itx][1].first);
        }
        LOG(INFO) << "Image " << itx << " finished";
      } else {
        // TODO: Change; inefficient
        correct_label[image_idx[itx]] =
              classifier_.get_label_index(tmp_predictions[itx][0].first);
        top_label_prob[itx].second = tmp_predictions[itx][0].second;
      }
      ++iterations[image_idx[itx]];
      ++itx;
    } // while loop end

    images_processed += finished_idx.size();

    if (finished_idx.size() > 0) {
      RemoveImages(finished_idx, x, image_idx, top_label_prob,
                  top_classifier_index, top_N_classifiers_index, image_size);

      if (data->shape(0) > x->shape(0) + images_processed) {
        size_t data_remaining = data->shape(0) - images_processed - x->shape(0);
        // min(data_remaining, data_removed)
        size_t data_to_add = std::min(data_remaining, finished_idx.size());
        size_t current_data_read = images_processed + x->shape(0);
        LOG(INFO) << current_data_read << " has been read from the data blob";
        AddImages(data_to_add, x, image_idx, top_label_prob,
                    top_classifier_index, top_N_classifiers_index, image_size, data, current_data_read,
                    current_index);
      }
    }

  } // while loop end
    // while loop end
  delete x;
  return r_final;
}

} // namespace caffe
