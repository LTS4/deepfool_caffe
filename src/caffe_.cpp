//
// caffe_.cpp provides wrappers of the caffe::Solver class, caffe::Net class,
// caffe::Layer class and caffe::Blob class and some caffe::Caffe functions,
// so that one could easily use Caffe from matlab.
// Note that for matlab, we will simply use float as the data type.

// Internally, data is stored with dimensions reversed from Caffe's:
// e.g., if the Caffe blob axes are (num, channels, height, width),
// the matcaffe data is stored as (width, height, channels, num)
// where width is the fastest dimension.

#include <sstream>
#include <string>
#include <vector>

#include "mex.h"

#include "caffe/caffe.hpp"
#include "caffe/util/deepfool.hpp"

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

using namespace caffe;  // NOLINT(build/namespaces)

// Do CHECK and throw a Mex error if check fails
inline void mxCHECK(bool expr, const char* msg) {
  if (!expr) {
    mexErrMsgTxt(msg);
  }
}
inline void mxERROR(const char* msg) { mexErrMsgTxt(msg); }

// Check if a file exists and can be opened
void mxCHECK_FILE_EXIST(const char* file) {
  std::ifstream f(file);
  if (!f.good()) {
    f.close();
    std::string msg("Could not open file ");
    msg += file;
    mxERROR(msg.c_str());
  }
  f.close();
}

// The pointers to caffe::Solver and caffe::Net instances
static vector<shared_ptr<Solver<float> > > solvers_;
static vector<shared_ptr<Net<float> > > nets_;
static vector<shared_ptr<DeepFool> > deepfool_nets_;
// init_key is generated at the beginning and everytime you call reset
static double init_key = static_cast<double>(caffe_rng_rand());

/** -----------------------------------------------------------------
 ** data conversion functions
 **/
// Enum indicates which blob memory to use
enum WhichMemory { DATA, DIFF };

// Copy matlab array to Blob data or diff
static void mx_mat_to_blob(const mxArray* mx_mat, Blob<float>* blob,
    WhichMemory data_or_diff) {
  mxCHECK(blob->count() == mxGetNumberOfElements(mx_mat),
      "number of elements in target blob doesn't match that in input mxArray");
  const float* mat_mem_ptr = reinterpret_cast<const float*>(mxGetData(mx_mat));
  float* blob_mem_ptr = NULL;
  switch (Caffe::mode()) {
  case Caffe::CPU:
    blob_mem_ptr = (data_or_diff == DATA ?
        blob->mutable_cpu_data() : blob->mutable_cpu_diff());
    break;
  case Caffe::GPU:
    blob_mem_ptr = (data_or_diff == DATA ?
        blob->mutable_gpu_data() : blob->mutable_gpu_diff());
    break;
  default:
    mxERROR("Unknown Caffe mode");
  }
  caffe_copy(blob->count(), mat_mem_ptr, blob_mem_ptr);
}

// Copy Blob data or diff to matlab array
static mxArray* blob_to_mx_mat(const Blob<float>* blob,
    WhichMemory data_or_diff) {
  const int num_axes = blob->num_axes();
  vector<mwSize> dims(num_axes);
  for (int blob_axis = 0, mat_axis = num_axes - 1; blob_axis < num_axes;
       ++blob_axis, --mat_axis) {
    dims[mat_axis] = static_cast<mwSize>(blob->shape(blob_axis));
  }
  // matlab array needs to have at least one dimension, convert scalar to 1-dim
  if (num_axes == 0) {
    dims.push_back(1);
  }
  mxArray* mx_mat =
      mxCreateNumericArray(dims.size(), dims.data(), mxSINGLE_CLASS, mxREAL);
  float* mat_mem_ptr = reinterpret_cast<float*>(mxGetData(mx_mat));
  const float* blob_mem_ptr = NULL;
  switch (Caffe::mode()) {
  case Caffe::CPU:
    blob_mem_ptr = (data_or_diff == DATA ? blob->cpu_data() : blob->cpu_diff());
    break;
  case Caffe::GPU:
    blob_mem_ptr = (data_or_diff == DATA ? blob->gpu_data() : blob->gpu_diff());
    break;
  default:
    mxERROR("Unknown Caffe mode");
  }
  caffe_copy(blob->count(), blob_mem_ptr, mat_mem_ptr);
  return mx_mat;
}

// Convert vector<int> to matlab row vector
static mxArray* int_vec_to_mx_vec(const vector<int>& int_vec) {
  mxArray* mx_vec = mxCreateDoubleMatrix(int_vec.size(), 1, mxREAL);
  double* vec_mem_ptr = mxGetPr(mx_vec);
  for (int i = 0; i < int_vec.size(); i++) {
    vec_mem_ptr[i] = static_cast<double>(int_vec[i]);
  }
  return mx_vec;
}

// Convert vector<string> to matlab cell vector of strings
static mxArray* str_vec_to_mx_strcell(const vector<std::string>& str_vec) {
  mxArray* mx_strcell = mxCreateCellMatrix(str_vec.size(), 1);
  for (int i = 0; i < str_vec.size(); i++) {
    mxSetCell(mx_strcell, i, mxCreateString(str_vec[i].c_str()));
  }
  return mx_strcell;
}

/** -----------------------------------------------------------------
 ** handle and pointer conversion functions
 ** a handle is a struct array with the following fields
 **   (uint64) ptr      : the pointer to the C++ object
 **   (double) init_key : caffe initialization key
 **/
// Convert a handle in matlab to a pointer in C++. Check if init_key matches
template <typename T>
static T* handle_to_ptr(const mxArray* mx_handle) {
  mxArray* mx_ptr = mxGetField(mx_handle, 0, "ptr");
  mxArray* mx_init_key = mxGetField(mx_handle, 0, "init_key");
  mxCHECK(mxIsUint64(mx_ptr), "pointer type must be uint64");
  mxCHECK(mxGetScalar(mx_init_key) == init_key,
      "Could not convert handle to pointer due to invalid init_key. "
      "The object might have been cleared.");
  return reinterpret_cast<T*>(*reinterpret_cast<uint64_t*>(mxGetData(mx_ptr)));
}

// Create a handle struct vector, without setting up each handle in it
template <typename T>
static mxArray* create_handle_vec(int ptr_num) {
  const int handle_field_num = 2;
  const char* handle_fields[handle_field_num] = { "ptr", "init_key" };
  return mxCreateStructMatrix(ptr_num, 1, handle_field_num, handle_fields);
}

// Set up a handle in a handle struct vector by its index
template <typename T>
static void setup_handle(const T* ptr, int index, mxArray* mx_handle_vec) {
  mxArray* mx_ptr = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
  *reinterpret_cast<uint64_t*>(mxGetData(mx_ptr)) =
      reinterpret_cast<uint64_t>(ptr);
  mxSetField(mx_handle_vec, index, "ptr", mx_ptr);
  mxSetField(mx_handle_vec, index, "init_key", mxCreateDoubleScalar(init_key));
}

// Convert a pointer in C++ to a handle in matlab
template <typename T>
static mxArray* ptr_to_handle(const T* ptr) {
  mxArray* mx_handle = create_handle_vec<T>(1);
  setup_handle(ptr, 0, mx_handle);
  return mx_handle;
}

// Convert a vector of shared_ptr in C++ to handle struct vector
template <typename T>
static mxArray* ptr_vec_to_handle_vec(const vector<shared_ptr<T> >& ptr_vec) {
  mxArray* mx_handle_vec = create_handle_vec<T>(ptr_vec.size());
  for (int i = 0; i < ptr_vec.size(); i++) {
    setup_handle(ptr_vec[i].get(), i, mx_handle_vec);
  }
  return mx_handle_vec;
}

/** -----------------------------------------------------------------
 ** matlab command functions: caffe_(api_command, arg1, arg2, ...)
 **/
// Usage: caffe_('get_solver', solver_file);
static void get_solver(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsChar(prhs[0]),
      "Usage: caffe_('get_solver', solver_file)");
  char* solver_file = mxArrayToString(prhs[0]);
  mxCHECK_FILE_EXIST(solver_file);
  SolverParameter solver_param;
  ReadSolverParamsFromTextFileOrDie(solver_file, &solver_param);
  shared_ptr<Solver<float> > solver(
      SolverRegistry<float>::CreateSolver(solver_param));
  solvers_.push_back(solver);
  plhs[0] = ptr_to_handle<Solver<float> >(solver.get());
  mxFree(solver_file);
}

// Usage: caffe_('solver_get_attr', hSolver)
static void solver_get_attr(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('solver_get_attr', hSolver)");
  Solver<float>* solver = handle_to_ptr<Solver<float> >(prhs[0]);
  const int solver_attr_num = 2;
  const char* solver_attrs[solver_attr_num] = { "hNet_net", "hNet_test_nets" };
  mxArray* mx_solver_attr = mxCreateStructMatrix(1, 1, solver_attr_num,
      solver_attrs);
  mxSetField(mx_solver_attr, 0, "hNet_net",
      ptr_to_handle<Net<float> >(solver->net().get()));
  mxSetField(mx_solver_attr, 0, "hNet_test_nets",
      ptr_vec_to_handle_vec<Net<float> >(solver->test_nets()));
  plhs[0] = mx_solver_attr;
}

// Usage: caffe_('solver_get_iter', hSolver)
static void solver_get_iter(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('solver_get_iter', hSolver)");
  Solver<float>* solver = handle_to_ptr<Solver<float> >(prhs[0]);
  plhs[0] = mxCreateDoubleScalar(solver->iter());
}

// Usage: caffe_('solver_restore', hSolver, snapshot_file)
static void solver_restore(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxIsChar(prhs[1]),
      "Usage: caffe_('solver_restore', hSolver, snapshot_file)");
  Solver<float>* solver = handle_to_ptr<Solver<float> >(prhs[0]);
  char* snapshot_file = mxArrayToString(prhs[1]);
  mxCHECK_FILE_EXIST(snapshot_file);
  solver->Restore(snapshot_file);
  mxFree(snapshot_file);
}

// Usage: caffe_('solver_solve', hSolver)
static void solver_solve(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('solver_solve', hSolver)");
  Solver<float>* solver = handle_to_ptr<Solver<float> >(prhs[0]);
  solver->Solve();
}

// Usage: caffe_('solver_step', hSolver, iters)
static void solver_step(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxIsDouble(prhs[1]),
      "Usage: caffe_('solver_step', hSolver, iters)");
  Solver<float>* solver = handle_to_ptr<Solver<float> >(prhs[0]);
  int iters = mxGetScalar(prhs[1]);
  solver->Step(iters);
}

// Usage: caffe_('get_net', model_file, phase_name)
static void get_net(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsChar(prhs[0]) && mxIsChar(prhs[1]),
      "Usage: caffe_('get_net', model_file, phase_name)");
  char* model_file = mxArrayToString(prhs[0]);
  char* phase_name = mxArrayToString(prhs[1]);
  mxCHECK_FILE_EXIST(model_file);
  Phase phase;
  if (strcmp(phase_name, "train") == 0) {
      phase = TRAIN;
  } else if (strcmp(phase_name, "test") == 0) {
      phase = TEST;
  } else {
    mxERROR("Unknown phase");
  }
  shared_ptr<Net<float> > net(new caffe::Net<float>(model_file, phase));
  nets_.push_back(net);
  plhs[0] = ptr_to_handle<Net<float> >(net.get());
  mxFree(model_file);
  mxFree(phase_name);
}

// Usage: caffe_('net_get_attr', hNet)
static void net_get_attr(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('net_get_attr', hNet)");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  const int net_attr_num = 6;
  const char* net_attrs[net_attr_num] = { "hLayer_layers", "hBlob_blobs",
      "input_blob_indices", "output_blob_indices", "layer_names", "blob_names"};
  mxArray* mx_net_attr = mxCreateStructMatrix(1, 1, net_attr_num,
      net_attrs);
  mxSetField(mx_net_attr, 0, "hLayer_layers",
      ptr_vec_to_handle_vec<Layer<float> >(net->layers()));
  mxSetField(mx_net_attr, 0, "hBlob_blobs",
      ptr_vec_to_handle_vec<Blob<float> >(net->blobs()));
  mxSetField(mx_net_attr, 0, "input_blob_indices",
      int_vec_to_mx_vec(net->input_blob_indices()));
  mxSetField(mx_net_attr, 0, "output_blob_indices",
      int_vec_to_mx_vec(net->output_blob_indices()));
  mxSetField(mx_net_attr, 0, "layer_names",
      str_vec_to_mx_strcell(net->layer_names()));
  mxSetField(mx_net_attr, 0, "blob_names",
      str_vec_to_mx_strcell(net->blob_names()));
  plhs[0] = mx_net_attr;
}

// Usage: caffe_('net_forward', hNet)
static void net_forward(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('net_forward', hNet)");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  net->ForwardPrefilled();
}

// Usage: caffe_('net_backward', hNet)
static void net_backward(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('net_backward', hNet)");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  net->Backward();
}

// Usage: caffe_('net_copy_from', hNet, weights_file)
static void net_copy_from(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxIsChar(prhs[1]),
      "Usage: caffe_('net_copy_from', hNet, weights_file)");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  char* weights_file = mxArrayToString(prhs[1]);
  mxCHECK_FILE_EXIST(weights_file);
  net->CopyTrainedLayersFrom(weights_file);
  mxFree(weights_file);
}

// Usage: caffe_('net_reshape', hNet)
static void net_reshape(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('net_reshape', hNet)");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  net->Reshape();
}

// Usage: caffe_('net_save', hNet, save_file)
static void net_save(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxIsChar(prhs[1]),
      "Usage: caffe_('net_save', hNet, save_file)");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  char* weights_file = mxArrayToString(prhs[1]);
  NetParameter net_param;
  net->ToProto(&net_param, false);
  WriteProtoToBinaryFile(net_param, weights_file);
  mxFree(weights_file);
}

// Usage: caffe_('layer_get_attr', hLayer)
static void layer_get_attr(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('layer_get_attr', hLayer)");
  Layer<float>* layer = handle_to_ptr<Layer<float> >(prhs[0]);
  const int layer_attr_num = 1;
  const char* layer_attrs[layer_attr_num] = { "hBlob_blobs" };
  mxArray* mx_layer_attr = mxCreateStructMatrix(1, 1, layer_attr_num,
      layer_attrs);
  mxSetField(mx_layer_attr, 0, "hBlob_blobs",
      ptr_vec_to_handle_vec<Blob<float> >(layer->blobs()));
  plhs[0] = mx_layer_attr;
}

// Usage: caffe_('layer_get_type', hLayer)
static void layer_get_type(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('layer_get_type', hLayer)");
  Layer<float>* layer = handle_to_ptr<Layer<float> >(prhs[0]);
  plhs[0] = mxCreateString(layer->type());
}

// Usage: caffe_('blob_get_shape', hBlob)
static void blob_get_shape(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('blob_get_shape', hBlob)");
  Blob<float>* blob = handle_to_ptr<Blob<float> >(prhs[0]);
  const int num_axes = blob->num_axes();
  mxArray* mx_shape = mxCreateDoubleMatrix(1, num_axes, mxREAL);
  double* shape_mem_mtr = mxGetPr(mx_shape);
  for (int blob_axis = 0, mat_axis = num_axes - 1; blob_axis < num_axes;
       ++blob_axis, --mat_axis) {
    shape_mem_mtr[mat_axis] = static_cast<double>(blob->shape(blob_axis));
  }
  plhs[0] = mx_shape;
}

// Usage: caffe_('blob_reshape', hBlob, new_shape)
static void blob_reshape(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxIsDouble(prhs[1]),
      "Usage: caffe_('blob_reshape', hBlob, new_shape)");
  Blob<float>* blob = handle_to_ptr<Blob<float> >(prhs[0]);
  const mxArray* mx_shape = prhs[1];
  double* shape_mem_mtr = mxGetPr(mx_shape);
  const int num_axes = mxGetNumberOfElements(mx_shape);
  vector<int> blob_shape(num_axes);
  for (int blob_axis = 0, mat_axis = num_axes - 1; blob_axis < num_axes;
       ++blob_axis, --mat_axis) {
    blob_shape[blob_axis] = static_cast<int>(shape_mem_mtr[mat_axis]);
  }
  blob->Reshape(blob_shape);
}

// Usage: caffe_('blob_get_data', hBlob)
static void blob_get_data(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('blob_get_data', hBlob)");
  Blob<float>* blob = handle_to_ptr<Blob<float> >(prhs[0]);
  plhs[0] = blob_to_mx_mat(blob, DATA);
}

// Usage: caffe_('blob_set_data', hBlob, new_data)
static void blob_set_data(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
      "Usage: caffe_('blob_set_data', hBlob, new_data)");
  Blob<float>* blob = handle_to_ptr<Blob<float> >(prhs[0]);
  mx_mat_to_blob(prhs[1], blob, DATA);
}

// Usage: caffe_('blob_get_diff', hBlob)
static void blob_get_diff(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('blob_get_diff', hBlob)");
  Blob<float>* blob = handle_to_ptr<Blob<float> >(prhs[0]);
  plhs[0] = blob_to_mx_mat(blob, DIFF);
}

// Usage: caffe_('blob_set_diff', hBlob, new_diff)
static void blob_set_diff(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
      "Usage: caffe_('blob_set_diff', hBlob, new_diff)");
  Blob<float>* blob = handle_to_ptr<Blob<float> >(prhs[0]);
  mx_mat_to_blob(prhs[1], blob, DIFF);
}

// Usage: caffe_('set_mode_cpu')
static void set_mode_cpu(MEX_ARGS) {
  mxCHECK(nrhs == 0, "Usage: caffe_('set_mode_cpu')");
  Caffe::set_mode(Caffe::CPU);
}

// Usage: caffe_('set_mode_gpu')
static void set_mode_gpu(MEX_ARGS) {
  mxCHECK(nrhs == 0, "Usage: caffe_('set_mode_gpu')");
  Caffe::set_mode(Caffe::GPU);
}

// Usage: caffe_('set_device', device_id)
static void set_device(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsDouble(prhs[0]),
      "Usage: caffe_('set_device', device_id)");
  int device_id = static_cast<int>(mxGetScalar(prhs[0]));
  Caffe::SetDevice(device_id);
}

// Usage: caffe_('get_init_key')
static void get_init_key(MEX_ARGS) {
  mxCHECK(nrhs == 0, "Usage: caffe_('get_init_key')");
  plhs[0] = mxCreateDoubleScalar(init_key);
}

// Usage: caffe_('reset')
static void reset(MEX_ARGS) {
  mxCHECK(nrhs == 0, "Usage: caffe_('reset')");
  // Clear solvers and stand-alone nets
  mexPrintf("Cleared %d solvers and %d stand-alone nets\n",
      solvers_.size(), nets_.size());
  solvers_.clear();
  nets_.clear();
  // Generate new init_key, so that handles created before becomes invalid
  init_key = static_cast<double>(caffe_rng_rand());
}

// Usage: caffe_('read_mean', mean_proto_file)
static void read_mean(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsChar(prhs[0]),
      "Usage: caffe_('read_mean', mean_proto_file)");
  char* mean_proto_file = mxArrayToString(prhs[0]);
  mxCHECK_FILE_EXIST(mean_proto_file);
  Blob<float> data_mean;
  BlobProto blob_proto;
  bool result = ReadProtoFromBinaryFile(mean_proto_file, &blob_proto);
  mxCHECK(result, "Could not read your mean file");
  data_mean.FromProto(blob_proto);
  plhs[0] = blob_to_mx_mat(&data_mean, DATA);
  mxFree(mean_proto_file);
}

// Usage: caffe_('write_mean', mean_data, mean_proto_file)
static void write_mean(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsSingle(prhs[0]) && mxIsChar(prhs[1]),
      "Usage: caffe_('write_mean', mean_data, mean_proto_file)");
  char* mean_proto_file = mxArrayToString(prhs[1]);
  int ndims = mxGetNumberOfDimensions(prhs[0]);
  mxCHECK(ndims >= 2 && ndims <= 3, "mean_data must have at 2 or 3 dimensions");
  const mwSize *dims = mxGetDimensions(prhs[0]);
  int width = dims[0];
  int height = dims[1];
  int channels;
  if (ndims == 3)
    channels = dims[2];
  else
    channels = 1;
  Blob<float> data_mean(1, channels, height, width);
  mx_mat_to_blob(prhs[0], &data_mean, DATA);
  BlobProto blob_proto;
  data_mean.ToProto(&blob_proto, false);
  WriteProtoToBinaryFile(blob_proto, mean_proto_file);
  mxFree(mean_proto_file);
}

// Usage: caffe_('get_deepfool', model_file, weights_file, labels_file)
static void get_deepfool(MEX_ARGS) {
  mxCHECK(nrhs == 3 && mxIsChar(prhs[0]) && mxIsChar(prhs[1])
          && mxIsChar(prhs[2]),
          "Usage: caffe_('get_deepfool', model_file, weights_file, labels_file)");
  // TODO: Check for memory problems?
  char* model_file = mxArrayToString(prhs[0]);
  char* weights_file = mxArrayToString(prhs[1]);
  char* labels_file = mxArrayToString(prhs[2]);
  mxCHECK_FILE_EXIST(model_file);
  mxCHECK_FILE_EXIST(weights_file);
  mxCHECK_FILE_EXIST(labels_file);
  shared_ptr<DeepFool> deepfool(new caffe::DeepFool(model_file, weights_file,
                                                    std::string(labels_file)));
  deepfool_nets_.push_back(deepfool);
  plhs[0] = ptr_to_handle<DeepFool>(deepfool.get());
  mxFree(model_file);
  mxFree(weights_file);
  mxFree(labels_file);
}

static void set_deepfool_parameters(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxIsStruct(prhs[1]),
          "Usage: caffe_('set_deepfool_parameters', )");

  DeepFool* df = handle_to_ptr<DeepFool>(prhs[0]);

  mxArray* num_of_labels = mxGetField(prhs[1], 0, "number_of_labels");
  if (num_of_labels) {
    df->set_num_of_labels(mxGetScalar(num_of_labels));
  }

  mxArray* max_iterations = mxGetField(prhs[1], 0, "max_iterations");
  if (max_iterations) {
    df->set_max_iterations(mxGetScalar(max_iterations));
  }

  mxArray* overshoot = mxGetField(prhs[1], 0, "overshoot");
  if (overshoot) {
    df->set_overshoot(mxGetScalar(overshoot));
  }

  mxArray* norm_p = mxGetField(prhs[1], 0, "norm_p");
  if (norm_p) {
    df->set_Q(mxGetScalar(norm_p));
  }

  mxArray* boundary_epsilon = mxGetField(prhs[1], 0, "boundary_epsilon");
  if (boundary_epsilon) {
    df->set_epsilon(mxGetScalar(boundary_epsilon));
  }
}

static void get_deepfool_parameters(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
          "Usage: caffe_('set_deepfool_parameters', )");

  DeepFool* df = handle_to_ptr<DeepFool>(prhs[0]);
  const int number_of_parameters = 5;
  const char* deepfool_params[number_of_parameters] = { "number_of_labels",
    "max_iterations", "overshoot", "norm_p", "boundary_epsilon" };

  mxArray* mx_deepfool_params = mxCreateStructMatrix(1, 1, number_of_parameters,
                                                     deepfool_params);

  mxSetField(mx_deepfool_params, 0, "number_of_labels",
             mxCreateDoubleScalar(df->get_num_of_labels()));
  mxSetField(mx_deepfool_params, 0, "max_iterations",
             mxCreateDoubleScalar(df->get_max_iterations()));
  mxSetField(mx_deepfool_params, 0, "overshoot",
             mxCreateDoubleScalar(df->get_overshoot()));
  mxSetField(mx_deepfool_params, 0, "norm_p",
             mxCreateDoubleScalar(df->get_Q()));
  mxSetField(mx_deepfool_params, 0, "boundary_epsilon",
             mxCreateDoubleScalar(df->get_epsilon()));

  plhs[0] = mx_deepfool_params;
}


// Usage: caffe_('deepfool_perturbation', deepfool, data)
static void deepfool_perturbation(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0])  && mxIsSingle(prhs[1]),
          "Usage: caffe_('deepfool_perturbation', deepfool, data)");

  mwSize input_dimensions_number = mxGetNumberOfDimensions(prhs[1]);

  mxCHECK(input_dimensions_number > 2 || input_dimensions_number < 5,
          "DeepFool usage: please provide a 3- or 4-dimensional data array");

  const mwSize* input_dimension = mxGetDimensions(prhs[1]);
  Blob<float>* data;
  if (input_dimensions_number == 3) {
    data = new Blob<float>(1, input_dimension[2],
                           input_dimension[1], input_dimension[0]);
  } else {
    data = new Blob<float>(input_dimension[3], input_dimension[2],
                           input_dimension[1], input_dimension[0]);
  }

  // sanity check
  mxCHECK(data->count() == mxGetNumberOfElements(prhs[1]),
          "The data blob and the number of data in the mxArray are different");

  // copy the data to the temporary input Blob; another alternative
  // is to copy the data to the networks input Blob but they might
  // not fit in memory in some circumstances
  const float* mat_mem_ptr = reinterpret_cast<const float*>(mxGetData(prhs[1]));
  caffe_copy(data->count(), mat_mem_ptr, data->mutable_cpu_data());

  DeepFool* df = handle_to_ptr<DeepFool>(prhs[0]);
  Blob<float>* output_data = new Blob<float>(data->shape(0), data->shape(1),
                                             data->shape(2), data->shape(3));
  std::vector<int> iterations(data->shape(0));
  std::vector<int> fooling_label(data->shape(0));
  std::vector<int> correct_label(data->shape(0));

  output_data = df->adversarial_perturbations(iterations, fooling_label,
                                              correct_label, data);

  // copy the output blob(s) of the algorithm and return them to MATLAB
  mxArray* output_mat =
      mxCreateNumericArray(input_dimensions_number, input_dimension,
                           mxSINGLE_CLASS, mxREAL);
  float* out_mat_mem_ptr = reinterpret_cast<float*>(mxGetData(output_mat));
  caffe_copy(output_data->count(), output_data->cpu_data(), out_mat_mem_ptr);

  // copy the rest of the algorithm output and return to MATLAB
  const mwSize elements[] = { data->shape(0) };

  mxArray* iterations_arr =
                      mxCreateNumericArray(1, elements, mxINT32_CLASS, mxREAL);
  int* itr_mat_mem_ptr = reinterpret_cast<int*>(mxGetData(iterations_arr));
  std::copy(iterations.begin(), iterations.end(),  itr_mat_mem_ptr);

  mxArray* fooling_label_arr =
                      mxCreateNumericArray(1, elements, mxINT32_CLASS, mxREAL);
  int* fool_lab_mat_mem_ptr =
                        reinterpret_cast<int*>(mxGetData(fooling_label_arr));
  std::copy(fooling_label.begin(), fooling_label.end(),  fool_lab_mat_mem_ptr);

  mxArray* correct_label_arr =
                      mxCreateNumericArray(1, elements, mxINT32_CLASS, mxREAL);
  int* corr_lab_mat_mem_ptr =
                        reinterpret_cast<int*>(mxGetData(correct_label_arr));
  std::copy(correct_label.begin(), correct_label.end(),  corr_lab_mat_mem_ptr);

  delete data;
  delete output_data;

  plhs[0] = output_mat;
  plhs[1] = iterations_arr;
  plhs[2] = fooling_label_arr;
  plhs[3] = correct_label_arr;
}

// Usage: caffe_('version')
static void version(MEX_ARGS) {
  mxCHECK(nrhs == 0, "Usage: caffe_('version')");
  // Return version string
  plhs[0] = mxCreateString(AS_STRING(CAFFE_VERSION));
}

/** -----------------------------------------------------------------
 ** Available commands.
 **/
struct handler_registry {
  string cmd;
  void (*func)(MEX_ARGS);
};

static handler_registry handlers[] = {
  // Public API functions
  { "get_solver",         get_solver      },
  { "solver_get_attr",    solver_get_attr },
  { "solver_get_iter",    solver_get_iter },
  { "solver_restore",     solver_restore  },
  { "solver_solve",       solver_solve    },
  { "solver_step",        solver_step     },
  { "get_net",            get_net         },
  { "net_get_attr",       net_get_attr    },
  { "net_forward",        net_forward     },
  { "net_backward",       net_backward    },
  { "net_copy_from",      net_copy_from   },
  { "net_reshape",        net_reshape     },
  { "net_save",           net_save        },
  { "layer_get_attr",     layer_get_attr  },
  { "layer_get_type",     layer_get_type  },
  { "blob_get_shape",     blob_get_shape  },
  { "blob_reshape",       blob_reshape    },
  { "blob_get_data",      blob_get_data   },
  { "blob_set_data",      blob_set_data   },
  { "blob_get_diff",      blob_get_diff   },
  { "blob_set_diff",      blob_set_diff   },
  { "set_mode_cpu",       set_mode_cpu    },
  { "set_mode_gpu",       set_mode_gpu    },
  { "set_device",         set_device      },
  { "get_init_key",       get_init_key    },
  { "reset",              reset           },
  { "read_mean",          read_mean       },
  { "write_mean",         write_mean      },
  { "version",            version         },
  { "get_deepfool",       get_deepfool    },
  { "deepfool_perturbation", deepfool_perturbation },
  { "get_deepfool_parameters", get_deepfool_parameters },
  { "set_deepfool_parameters", set_deepfool_parameters },
  // The end.
  { "END",                NULL            },
};

/** -----------------------------------------------------------------
 ** matlab entry point.
 **/
// Usage: caffe_(api_command, arg1, arg2, ...)
void mexFunction(MEX_ARGS) {
  mexLock();  // Avoid clearing the mex file.
  mxCHECK(nrhs > 0, "Usage: caffe_(api_command, arg1, arg2, ...)");
  // Handle input command
  char* cmd = mxArrayToString(prhs[0]);
  bool dispatched = false;
  // Dispatch to cmd handler
  for (int i = 0; handlers[i].func != NULL; i++) {
    if (handlers[i].cmd.compare(cmd) == 0) {
      handlers[i].func(nlhs, plhs, nrhs-1, prhs+1);
      dispatched = true;
      break;
    }
  }
  if (!dispatched) {
    ostringstream error_msg;
    error_msg << "Unknown command '" << cmd << "'" ;
    mxERROR(error_msg.str().c_str());
  }
  mxFree(cmd);
}
