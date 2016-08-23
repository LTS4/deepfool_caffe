classdef DeepFool < handle
  % Wrapper class of caffe::DeepFool in matlab

  properties (Access = private)
    hDeepFool_self
  end

  properties (SetAccess = private)
    number_of_labels
    overshoot
    max_iterations
    norm_p
    epsilon
  end % properties

  methods
    function self = DeepFool(varargin)
      % Use one of the following alternatives to create a DeepFool object:
      %   (1) df = DeepFool(model_file, weigths, labels_file)

      % if a handler has been provided
      if ~(nargin == 1 && isstruct(varargin{1}))
        self = caffe.get_deepfool(varargin{:});
        return
      end

      hDeepFool_self = varargin{1};
      CHECK(is_valid_handle(hDeepFool_self), 'invalid DeepFool handle');

      self.hDeepFool_self = hDeepFool_self;
      get_algorithm_options(self);
    end % constructor

    function [perturbations, iterations, fooling_label, true_label] = ...
                                    adversarial_perturbations(self, input_data)
      CHECK(isfloat(input_data), 'input_data must be a float array');
      [perturbations, iterations, fooling_label, true_label] = ...
              caffe_('deepfool_perturbation', self.hDeepFool_self, input_data);
    end

    % set the specified parameters (options) and also set the
    % relevant class properties; if some parameters are not
    % specified take their current value
    function set_algorithm_options(self, opts)
      CHECK(isstruct(opts), ['Please pass a struct to define the ', ...
            'arguments that you desire, like:', ...
            'opts.max_iterations = 100', ...
            'opts.overshoot = 0.2', ...
            'DeepFoolObject.set_algorithm_options(opts)']);

      if isfield(opts, 'number_of_labels')
        self.number_of_labels = opts.number_of_labels;
      else
        opts.number_of_labels = self.number_of_labels;
      end
      if isfield(opts, 'max_iterations')
        self.max_iterations = opts.max_iterations;
      else
        opts.max_iterations = self.max_iterations;
      end
      if isfield(opts, 'overshoot')
        self.overshoot = opts.overshoot;
      else
        opts.overshoot = self.overshoot;
      end
      if isfield(opts, 'norm_p')
        self.norm_p = opts.norm_p;
      else
        otps.norm_p = self.norm_p;
      end
      if isfield(opts, 'epsilon')
        self.epsilon = opts.epsilon;
      else
        opts.epsilon = self.epsilon;
      end

      caffe_('set_deepfool_parameters', self.hDeepFool_self, opts);
    end

    % get the current parameters (options) and also set the
    % relevant class properties
    function [opts] = get_algorithm_options(self)
      opts = caffe_('get_deepfool_parameters', self.hDeepFool_self);
      self.number_of_labels = opts.number_of_labels;
      self.max_iterations = opts.max_iterations;
      self.overshoot = opts.overshoot;
      self.norm_p = opts.norm_p;
      self.epsilon = opts.epsilon;
    end
  end % methods
end % classdef
