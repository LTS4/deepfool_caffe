#ifndef CAFFE_DEEPFOOL_HPP_
#define CAFFE_DEEPFOOL_HPP_

#include <string>

#include <caffe/util/classifier.hpp>

namespace caffe {
class DeepFool {
 public:

    // TODO: Provide more constructors?
    DeepFool(const std::string& model_file,
             const std::string& weights_file,
             const std::string& label_file = "",
             const std::string& mean_file = "",
             const size_t number_of_labels = 10,
             const size_t batch_size = 128,
             const int MAX_ITER = 50,
             const int Q = 2,
             const float OVERSHOOT = 0.02);

    Blob<float>* adversarial_perturbations(Blob<float>* data = NULL);
    Blob<float>* adversarial_perturbations(std::vector<int>& iterations,
                                      std::vector<int>& fooling_label,
                                      std::vector<int>& correct_label,
                                      Blob<float>* data = NULL);

    /**
     *  @brief Compute the adversarial images for a image in OpenCV's
     *         Mat format.
     *
     */
    void adversarial(cv::Mat image, bool PREPROCESSING=false);

    /**
     *  @brief Compute the adversarial images for a vector that contains
     *         images in OpenCV's Mat format.
     */
    void adversarial(std::vector<cv::Mat> images, bool PREPROCESSING=false);

    /**
     *  @brief Preprocess a single image. WARNING: The object needs to have
     *         a classifier specified before this function is used.
     *
     *  TODO: Test if the classifier has been initialized
     */
    void Preprocess(cv::Mat& image) { classifier_.Preprocess(image); }


    /**
     *  @brief Preprocess a vector of images. WARNING: The object needs
     *         to have a classifier specified before this function is used.
     *
     *  TODO: Test if the classifier has been initialized
     */
    void Preprocess(std::vector<cv::Mat>& images) {
      classifier_.Preprocess(images);
    }

    // get functions for the algorithms parameters
    inline int get_max_iterations() const { return max_iterations_; }
    inline int get_Q() const { return Q_; }
    inline size_t get_batch_size() const { return batch_size_; }
    inline size_t get_num_of_labels() const { return number_of_labels_; }
    inline float get_overshoot() const { return overshoot_; }
    inline float get_epsilon() const { return epsilon_; }

    // set functions for the algorithms parameters
    inline void set_max_iterations(int max_iterations) {
      max_iterations_ = max_iterations;
    }
    inline void set_Q(int Q) {
      Q_ = Q;
    }
    inline void set_overshoot(float overshoot) {
      overshoot_ = overshoot;
    }
    inline void set_batch_size(size_t batch_size) {
      batch_size_ = batch_size;
    }
    inline void set_num_of_labels(size_t number_of_labels) {
      number_of_labels_ = number_of_labels;
    }

    inline void set_file_names(std::vector<std::string> file_names) {
      file_names_ = file_names;
    }

    inline void set_epsilon(float epsilon) {
      epsilon_ = epsilon;
    }

    void ImportData(std::vector<cv::Mat> data, bool RESHAPE=true);

 private:
    Classifier classifier_;
    // number_of_labels_ defines the number of the top classifier that will be
    // used to find the minimum perturbation; it is a (good) aproximation
    //  of the original algorithm which is used to speed up the process
    size_t number_of_labels_;
    int max_iterations_;
    int Q_;
    float overshoot_;
    float epsilon_;

    // batch_size_ defines the maximum number of images that will
    // whose perturbations will be computed simultaneously to avoid
    // memory problems; used only when the input is a set of images
    size_t batch_size_;

    std::vector<std::string> file_names_;

    void reshapeGradient(const std::vector<float>&, caffe::Blob<float>*);

    void RemoveImages(const vector<int>& finished_idx,
                      Blob<float>* data, vector<int>& image_idx,
                      vector<Classifier::Prediction>& top_label_prob,
                      vector<int >& top_classifier_index,
                      vector<vector<int> >& top_10_index,
                      size_t image_size);

    void AddImages(size_t data_to_add, Blob<float>* x, vector<int>& image_idx,
                   vector<Classifier::Prediction>& top_label_prob,
                   vector<int>& top_classifier_index,
                   vector<vector<int> >& top_10_index,
                   size_t image_size, Blob<float>* data,
                   size_t current_data_read,
                   size_t& current_index);

    std::string print_predictions(std::vector<std::vector<Classifier::Prediction> > predictions,
                                  std::vector<int> image_idx, size_t topN);
}; // class DeepFool


Blob<float>* cv_mat_to_blob(std::vector<cv::Mat> data);

} // namespace caffe

#endif // CAFFE_DEEPFOOL_HPP_
