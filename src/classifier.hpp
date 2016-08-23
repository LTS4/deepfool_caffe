#ifndef CAFFE_CLASSIFIER_HPP_
#define CAFFE_CLASSIFIER_HPP_

// STL libraries
#include <string>
#include <vector>
#include <utility> // for std::pair
#include <memory>  // for shared_ptr but using boost's implementation

// other libraries
// TODO: Use the same #ifdef at the functions
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif // USE_OPENCV

#include <boost/shared_ptr.hpp>

// caffe header
#include <caffe/caffe.hpp>

namespace caffe {

class Classifier {
 public:

  // datatype that is returned from the Classifier after the classification
  typedef std::pair<std::string, float> Prediction;

  Classifier(const std::string& model_file,
             const std::string& trained_file,
             const std::string& label_file = "",
             const std::string& mean_file = "");

  // TODO: Define the constructor
  Classifier(boost::shared_ptr<caffe::Net<float> > net,
             const std::string& label_file = "",
             const std::string& mean_file = "");

  /**
   * @brief Classify the images that are defined in the data blob (4D matrix)
   *        and return the top-N labels alongside their predictions (or the
   *        network's output, in general).
   *
   * TODO: IMPLEMENT IT -- maybe just change the other Classify function
   */
  std::vector<std::vector<Prediction> >
  Classify(const Blob<float>* data, int N = 5);

  /**
   *  @brief Classify a cv::Mat object and return the top-N labels
   */
  std::vector<std::vector<Prediction> >
  Classify(const cv::Mat& img, int N = 5);

  /**
    * @brief Classify the image and return the top-N labels alongside their
    *        predictions (or the network's output, in general).
    *
    */
  std::vector<std::vector<Prediction> >
  Classify(const std::vector<cv::Mat>& img, int N = 5);

  /**
   * @brief Classify whatever data happens to be in the input blob and return
   *        their top-N labels alongside their predictions. Use
   *        Classifier::ImportData() to import the data of your preference
   *        in the input layer.
   */
  std::vector<std::vector<Prediction> > Classify(int N = 5);

  /**
   * @brief Get the network's output (predictions) given a blob that contains
   *        the data.
   */
  std::vector<float> Predict(const Blob<float>* data);

  /**
   * @brief Get the network's output (predictions) for the data that are
   *        already contained in the input layer.
   */
  std::vector<std::vector<float> > Predict();

  /**
   * @brief Get the netwok's output (predictions) given the img image file;
   *        supports all the file types that are supported from the OpenCV
   *        library.
   *
   * TODO: Make it work with a vector<cv::Mat>
   */
  std::vector<std::vector<float> > Predict(const std::vector<cv::Mat>& img);

  std::vector<std::vector<float> > Predict(const cv::Mat& img);


  /**
   * @brief Get the gradient with respect to the input of the k-th classifier
   *
   * TODO: Add more detailed description
   */
  std::vector<std::vector<float> >
  InputGradientofClassifier(const std::vector<cv::Mat>& img, int k = 0);

  std::vector<std::vector<float> >
  InputGradientofClassifier(const cv::Mat& img, int k =0);

  std::vector<std::vector<float> > InputGradientofClassifier(int k = 0);

  std::vector<std::vector<float> > InputGradientofClassifier2(vector<int> k);

  /**
   * @brief Preprocess the given cv::Mat image. It is best to have the
   *        image in a CV_8UC3 or CV_8UC1 representation. This function
   *        will transform it to the appropriate CV_32FC3 or CV_32FC1
   *        format. Passing a CV_32FC3 or CV_32FC1 image before yields
   *        slightly different results.
   *
   * TODO: Add similar functionality for images already represented as
   *       Blob<floats> ?
   *
   */
  void Preprocess(cv::Mat& img);

  /**
    * @brief Preprocess a vector of cv::Mat images. It is best to have the
    *        image in a CV_8UC3 or CV_8UC1 representation. This function
    *        will transform it to the appropriate CV_32FC3 or CV_32FC1
    *        format. Passing a CV_32FC3 or CV_32FC1 image before yields
    *        slightly different results.
    */
  void Preprocess(std::vector<cv::Mat>& data);

  // TODO: Implement the second function
  void ImportData(const std::vector<cv::Mat>& data, bool RESHAPE=true);
  void ImportData(const caffe::Blob<float>* data, bool RESHAPE=true);

  /**
   * @brief Get the layer names of the defined network.
   */
  inline std::vector<string> get_layer_names() const {
    return net_->layer_names();
  }

  /**
   * @brief Get the labels that the network can discriminate.
   */
  inline std::vector<string> get_labels() const { return labels_; }

  /**
   * @brief Get the geometry of the input data as a cv::Size object.
   *
   * TODO: MAYBE improve the function
   */
  inline cv::Size get_geometry() const { return input_geometry_; }

  // TODO: Just a testing function -- REMOVE
  float print_mean_file(int i, int j) {
    return mean_.at<float>(i,j);
  }

  // TODO: Just a testing function -- REMOVE
  void save_mean_as_image() {
    cv::imwrite("mean_image.jpeg", mean_);
  }

  size_t get_label_index(std::string label_name);

  /**
   * @brief Get the mean image used for preprocessing
   *
   */
  inline cv::Mat get_mean() const { return mean_; }

  inline Blob<float>* get_input_blob() {
    return net_->input_blobs()[0];
  }

 private:
  // pointer for the underlaying network
  boost::shared_ptr<caffe::Net<float> > net_;
  // size of the input data
  std::vector<int> input_size_;
  // the labels that are related to the specific network
  std::vector<std::string> labels_;

  // TODO: Use USE_OPENCV to avoid problems?
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;

  // method to set the mean_ variable used for preprocessing the cv::Mat images
  void SetMean(const std::string& mean_file);

  // TODO: REMOVE at a later stage
  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  /**
   * Used for data preprocessing
   * TODO: Remove this function
   */
  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

  std::vector<std::vector<float> > GetNetworkOutput();

  void LabelsInit(const std::string& label_file,
                          const std::string& mean_file);

};

// related functions for reading images from the directory
cv::Mat read_image(std::string image_file);//, bool UINT8_TO_FLOAT32=false);
std::vector<cv::Mat> read_images_from_dir(std::string dir_name);
std::vector<std::string> read_names_from_dir(std::string dir_name);
void DefineMode(const std::string& gpu_selection);
} // namespace Caffe

#endif // CAFFE_CLASSIFIER_HPP_
