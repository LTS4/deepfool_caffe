function deepfool = get_deepfool(varargin)
% Add comments

CHECK(nargin == 3, ['usage: ', ...
  'deepfool = get_deepfool(model_file, weights_file, labels_file)']);


model_file = varargin{1};
weights_file = varargin{2};
labels_file = varargin{3};

CHECK(ischar(model_file), 'model_file must be a string');
CHECK_FILE_EXIST(model_file);
CHECK(ischar(weights_file), 'weights_file must be a string');
CHECK_FILE_EXIST(weights_file);
CHECK(ischar(labels_file), 'labels_file must be a string');
CHECK_FILE_EXIST(labels_file);

hDeepFool = caffe_('get_deepfool', model_file, weights_file, labels_file);
deepfool = caffe.DeepFool(hDeepFool);

end
