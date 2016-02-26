local ffi = require 'ffi'

local THNN = {}

local generic_THNN_h = [[
TH_API void THNN_(Abs_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output);
TH_API void THNN_(Abs_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput);

TH_API void THNN_(AbsCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage);
TH_API void THNN_(AbsCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage);

TH_API void THNN_(ClassNLLCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *output,
          bool sizeAverage,
          THTensor *weights,
          THTensor *total_weight);
TH_API void THNN_(ClassNLLCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *gradInput,
          bool sizeAverage,
          THTensor *weights,
          THTensor *total_weight);

TH_API void THNN_(ELU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real alpha);
TH_API void THNN_(ELU_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output,
          real alpha);

TH_API void THNN_(DistKLDivCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage);
TH_API void THNN_(DistKLDivCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage);

TH_API void THNN_(HardShrink_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real lambda);
TH_API void THNN_(HardShrink_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          real lambda);

TH_API void THNN_(HardTanh_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real min_val,
          real max_val);
TH_API void THNN_(HardTanh_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          real min_val,
          real max_val);

TH_API void THNN_(L1Cost_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output);
TH_API void THNN_(L1Cost_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput);

TH_API void THNN_(LeakyReLU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real negval,
          bool inplace);
TH_API void THNN_(LeakyReLU_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          real negval,
          bool inplace);

TH_API void THNN_(LogSigmoid_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *buffer);
TH_API void THNN_(LogSigmoid_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *buffer);

TH_API void THNN_(LogSoftMax_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output);
TH_API void THNN_(LogSoftMax_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output);

TH_API void THNN_(LookupTable_accGradParameters)(
          THNNState *state,
          THIndexTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THIntegerTensor *count,
          THTensor *sorted,
          THTensor *indices,
          bool scaleGradByFreq,
          int paddingValue,
          real scale);

TH_API void THNN_(MarginCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage,
          real margin);
TH_API void THNN_(MarginCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage,
          real margin);

TH_API void THNN_(MSECriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage);
TH_API void THNN_(MSECriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage);

TH_API void THNN_(MultiLabelMarginCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage);
TH_API void THNN_(MultiLabelMarginCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage);

TH_API void THNN_(MultiMarginCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage,
          int p,
          THTensor *weights);
TH_API void THNN_(MultiMarginCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage,
          int p,
          THTensor *weights);

TH_API void THNN_(PReLU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THIndex_t nOutputPlane);
TH_API void THNN_(PReLU_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THIndex_t nOutputPlane);
TH_API void THNN_(PReLU_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *gradWeight,
          THTensor *gradWeightBuf,
          THTensor *gradWeightBuf2,
          THIndex_t nOutputPlane,
          real scale);

TH_API void THNN_(RReLU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *noise,
          real lower,
          real upper,
          bool train,
          bool inplace,
          THGenerator *generator);
TH_API void THNN_(RReLU_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *noise,
          real lower,
          real upper,
          bool train,
          bool inplace);

TH_API void THNN_(Sigmoid_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output);
TH_API void THNN_(Sigmoid_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output);

TH_API void THNN_(SmoothL1Criterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage);
TH_API void THNN_(SmoothL1Criterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage);

TH_API void THNN_(SoftMax_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output);
TH_API void THNN_(SoftMax_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output);

TH_API void THNN_(SoftPlus_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real beta,
          real threshold);
TH_API void THNN_(SoftPlus_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output,
          real beta,
          real threshold);

TH_API void THNN_(SoftShrink_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real lambda);
TH_API void THNN_(SoftShrink_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          real lambda);

TH_API void THNN_(SparseLinear_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *cudaBuffer,
          THTensor *shardBuffer);
TH_API void THNN_(SparseLinear_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight);
TH_API void THNN_(SparseLinear_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *weight,
          THTensor *bias,
          real weightDecay,
          real scale);
TH_API void THNN_(SparseLinear_zeroGradParameters)(
          THNNState *state,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *lastInput);
TH_API void THNN_(SparseLinear_updateParameters)(
          THNNState *state,
          THTensor *weight,
          THTensor *bias,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *lastInput,
          real learningRate);

TH_API void THNN_(Sqrt_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real eps);
TH_API void THNN_(Sqrt_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output);

TH_API void THNN_(Square_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output);
TH_API void THNN_(Square_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput);

TH_API void THNN_(Tanh_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output);
TH_API void THNN_(Tanh_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output);

TH_API void THNN_(Threshold_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real threshold,
          real val,
          bool inplace);
TH_API void THNN_(Threshold_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          real threshold,
          bool inplace);

TH_API void THNN_(TemporalConvolution_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          int kW, int dW,
          int inputFrameSize,
          int outputFrameSize);

TH_API void THNN_(TemporalConvolution_updateGradInput)(
          THNNState* state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          int kW, int dW);

TH_API void THNN_(TemporalConvolution_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          int kW, int dW,
          real scale);

TH_API void THNN_(TemporalMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *indices,
          int kW, int dW);

TH_API void THNN_(TemporalMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *indices,
          int kW, int dW);

TH_API void THNN_(TemporalSubSampling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          int kW, int dW,
          int inputFrameSize);

TH_API void THNN_(TemporalSubSampling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          int kW, int dW);

TH_API void THNN_(TemporalSubSampling_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          int kW, int dW,
          real scale);

TH_API void THNN_(BatchNormalization_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *running_mean,
          THTensor *running_var,
          THTensor *save_mean,
          THTensor *save_std,
          bool train,
          double momentum,
          double eps);
TH_API void THNN_(BatchNormalization_backward)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *weight,
          THTensor *save_mean,
          THTensor *save_std,
          double scale);

TH_API void THNN_(SpatialConvolutionMap_updateOutput)(
          THNNState *state,       // library state
          THTensor *input,        // input tensor
          THTensor *output,       // [OUT] convolution output
          THTensor *weight,       // 3D weight tensor (connTable:size(1) x kH x kW)
          THTensor *bias,         // 1D bias tensor (nOutputPlane)
          THTensor *connTable,    // connection table
          int nInputPlane,        // number of input planes
          int nOutputPlane,       // number of output planes
          int dW, int dH);        // stride
TH_API void THNN_(SpatialConvolutionMap_updateGradInput)(
          THNNState *state,       // library state
          THTensor *input,        // input tensor
          THTensor *gradOutput,   // gradient w.r.t. output
          THTensor *gradInput,    // [OUT] gradient w.r.t. input
          THTensor *weight,       // 3D weight tensor (connTable:size(1) x kH x kW)
          THTensor *bias,         // 1D bias tensor (nOutputPlane)
          THTensor *connTable,    // connection table
          int nInputPlane,        // number of input planes
          int nOutputPlane,       // number of output planes
          int dW, int dH);        // stride
TH_API void THNN_(SpatialConvolutionMap_accGradParameters)(
          THNNState *state,       // library state
          THTensor *input,        // input tensor
          THTensor *gradOutput,   // gradient w.r.t. output
          THTensor *gradWeight,   // 3D gradWeight tensor (connTable:size(1) x kH x kW)
          THTensor *gradBias,     // 1D gradBias tensor (nOutputPlane)
          THTensor *connTable,    // connection table
          int nInputPlane,        // number of input planes
          int nOutputPlane,       // number of output planes
          int dW, int dH,         // stride
          real scale);            // scaling factor

TH_API void THNN_(SpatialConvolutionMM_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          THTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH);
TH_API void THNN_(SpatialConvolutionMM_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          THTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH);
TH_API void THNN_(SpatialConvolutionMM_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,
          THTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          real scale);

TH_API void THNN_(SpatialConvolutionLocal_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          THTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          long inputWidth, long inputHeight,
          long outputWidth, long outputHeight);
TH_API void THNN_(SpatialConvolutionLocal_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput,
          THTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          long inputWidth, long inputHeight,
          long outputWidth, long outputHeight);
TH_API void THNN_(SpatialConvolutionLocal_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,
          THTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          long inputWidth, long inputHeight,
          long outputWidth, long outputHeight,
          real scale);

TH_API void THNN_(SpatialAdaptiveMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *indices,
          int owidth, int oheight);
TH_API void THNN_(SpatialAdaptiveMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *indices);

TH_API void THNN_(SpatialAveragePooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          bool ceil_mode,
          bool count_include_pad);
TH_API void THNN_(SpatialAveragePooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          bool ceil_mode,
          bool count_include_pad);

TH_API void THNN_(SpatialFractionalMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int outputW, int outputH,
          int poolSizeW, int poolSizeH,
          THTensor *indices,
          THTensor *randomSamples);
TH_API void THNN_(SpatialFractionalMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int outputW, int outputH,
          int poolSizeW, int poolSizeH,
          THTensor *indices);

TH_API void THNN_(SpatialFullConvolution_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *columns,
          THTensor *ones,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int adjW, int adjH);
TH_API void THNN_(SpatialFullConvolution_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *gradColumns,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int adjW, int adjH);
TH_API void THNN_(SpatialFullConvolution_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *columns,
          THTensor *ones,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int adjW, int adjH,
          real scale);

TH_API void THNN_(SpatialFullConvolutionMap_updateOutput)(
          THNNState *state,       // library state
          THTensor *input,        // input tensor
          THTensor *output,       // [OUT] convolution output
          THTensor *weight,       // 3D weight tensor (connTable:size(1) x kH x kW)
          THTensor *bias,         // 1D bias tensor (nOutputPlane)
          THTensor *connTable,    // connection table
          int nInputPlane,        // number of input planes
          int nOutputPlane,       // number of output planes
          int dW, int dH);        // stride
TH_API void THNN_(SpatialFullConvolutionMap_updateGradInput)(
          THNNState *state,       // library state
          THTensor *input,        // input tensor
          THTensor *gradOutput,   // gradient w.r.t. output
          THTensor *gradInput,    // [OUT] gradient w.r.t. input
          THTensor *weight,       // 3D weight tensor (connTable:size(1) x kH x kW)
          THTensor *bias,         // 1D bias tensor (nOutputPlane)
          THTensor *connTable,    // connection table
          int nInputPlane,        // number of input planes
          int nOutputPlane,       // number of output planes
          int dW, int dH);        // stride
TH_API void THNN_(SpatialFullConvolutionMap_accGradParameters)(
          THNNState *state,       // library state
          THTensor *input,        // input tensor
          THTensor *gradOutput,   // gradient w.r.t. output
          THTensor *gradWeight,   // 3D gradWeight tensor (connTable:size(1) x kH x kW)
          THTensor *gradBias,     // 1D gradBias tensor (nOutputPlane)
          THTensor *connTable,    // connection table
          int nInputPlane,        // number of input planes
          int nOutputPlane,       // number of output planes
          int dW, int dH,         // stride
          real scale);            // scaling factor

TH_API void THNN_(SpatialMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *indices,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          bool ceil_mode);
TH_API void THNN_(SpatialMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *indices,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          bool ceil_mode);

TH_API void THNN_(SpatialMaxUnpooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *indices,
          int owidth, int oheight);
TH_API void THNN_(SpatialMaxUnpooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *indices,
          int owidth, int oheight);

TH_API void THNN_(SpatialSubSampling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          int kW, int kH,
          int dW, int dH);
TH_API void THNN_(SpatialSubSampling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          int kW, int kH,
          int dW, int dH);
TH_API void THNN_(SpatialSubSampling_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          int kW, int kH,
          int dW, int dH,
          real scale);

TH_API void THNN_(SpatialUpSamplingNearest_updateOutput)(
        THNNState *state,
        THTensor *input,
        THTensor *output,
        int scale_factor);
TH_API void THNN_(SpatialUpSamplingNearest_updateGradInput)(
        THNNState *state,
        THTensor *input,
        THTensor *gradOutput,
        THTensor *gradInput,
        int scale_factor);

TH_API void THNN_(VolumetricAveragePooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int kT, int kW, int kH,
          int dT, int dW, int dH);
TH_API void THNN_(VolumetricAveragePooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int kT, int kW, int kH,
          int dT, int dW, int dH);

TH_API void THNN_(VolumetricConvolution_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          THTensor *fgradInput,
          int dT, int dW, int dH,
          int pT, int pW, int pH);
TH_API void THNN_(VolumetricConvolution_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput,
          int dT, int dW, int dH,
          int pT, int pW, int pH);
TH_API void THNN_(VolumetricConvolution_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,
          THTensor *fgradInput,
          int dT, int dW, int dH,
          int pT, int pW, int pH,
          real scale);

TH_API void THNN_(VolumetricConvolutionMM_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int pT, int pW, int pH);
TH_API void THNN_(VolumetricConvolutionMM_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput,
          THTensor *fgradInput,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int pT, int pW, int pH);
TH_API void THNN_(VolumetricConvolutionMM_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,
          real scale);

TH_API void THNN_(VolumetricFullConvolution_updateOutput)(
          THNNState *state,         // library state
          THTensor *input,          // 4D or 5D (batch) tensor
          THTensor *output,         // [OUT] volumetric convolution output
          THTensor *weight,         // weight tensor (nInputPlane x nOutputPlane x kT x kH x kW)
          THTensor *bias,           // gradBias tensor (nOutputPlane)
          THTensor *finput,         // [OUT] internal columns buffer
          THTensor *fgradInput,     // [OUT] internal ones buffer
          int dT, int dW, int dH,   // stride of the convolution
          int pT, int pW, int pH,   // padding
          int aT, int aW, int aH);  // extra output adjustment
TH_API void THNN_(VolumetricFullConvolution_updateGradInput)(
          THNNState *state,         // library state
          THTensor *input,          // 4D or 5D (batch) tensor
          THTensor *gradOutput,     // gradient w.r.t. output
          THTensor *gradInput,      // [OUT] gradient w.r.t. input
          THTensor *weight,         // weight tensor (nInputPlane x nOutputPlane x kT x kH x kW)
          THTensor *finput,         // internal columns buffer
          THTensor *fgradInput,     // internal ones buffer
          int dT, int dW, int dH,   // stride
          int pT, int pW, int pH,   // padding
          int aT, int aW, int aH);  // extra output adjustment
TH_API void THNN_(VolumetricFullConvolution_accGradParameters)(
          THNNState *state,         // library state
          THTensor *input,          // 4D or 5D (batch) tensor
          THTensor *gradOutput,     // gradient w.r.t. output
          THTensor *gradWeight,     // gradWeight tensor (nInputPlane x nOutputPlane x kT x kH x kW)
          THTensor *gradBias,       // gradBias tensor (nOutputPlane)
          THTensor *finput,         // internal columns buffer
          THTensor *fgradInput,     // internal ones buffer
          int dT, int dW, int dH,   // stride
          int pT, int pW, int pH,   // padding
          int aT, int aW, int aH,   // extra output adjustment
          real scale);              // scaling factor

TH_API void THNN_(VolumetricMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *indices,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int pT, int pW, int pH,
          bool ceilMode);
TH_API void THNN_(VolumetricMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *indices,
          int dT, int dW, int dH,
          int pT, int pW, int pH);

TH_API void THNN_(VolumetricMaxUnpooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *indices,
          int oT, int oW, int oH,
          int dT, int dW, int dH,
          int pT, int pW, int pH);
TH_API void THNN_(VolumetricMaxUnpooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *indices,
          int oT, int oW, int oH,
          int dT, int dW, int dH,
          int pT, int pW, int pH);

TH_API void THNN_(SpatialReflectionPadding_updateOutput)(THNNState *state,
                                                         THTensor *input,
                                                         THTensor *output,
                                                         int pad_l, int pad_r,
                                                         int pad_t, int pad_b);

TH_API void THNN_(SpatialReflectionPadding_updateGradInput)(THNNState *state,
                                                            THTensor *input,
                                                            THTensor *gradOutput,
                                                            THTensor *gradInput,
                                                            int pad_l, int pad_r,
                                                            int pad_t, int pad_b);

TH_API void THNN_(SpatialReplicationPadding_updateOutput)(THNNState *state,
                                                         THTensor *input,
                                                         THTensor *output,
                                                         int pad_l, int pad_r,
                                                         int pad_t, int pad_b);

TH_API void THNN_(SpatialReplicationPadding_updateGradInput)(THNNState *state,
                                                            THTensor *input,
                                                            THTensor *gradOutput,
                                                            THTensor *gradInput,
                                                            int pad_l, int pad_r,
                                                            int pad_t, int pad_b);

]]

-- THGenerator struct declaration copied from torch7/lib/TH/THRandom.h
local base_declarations = [[
typedef void THNNState;

typedef struct {
  unsigned long the_initial_seed;
  int left;
  int seeded;
  unsigned long next;
  unsigned long state[624]; /* the array for the state vector 624 = _MERSENNE_STATE_N  */
  double normal_x;
  double normal_y;
  double normal_rho;
  int normal_is_valid;
} THGenerator;
]]

-- polyfill for LUA 5.1
if not package.searchpath then
   local sep = package.config:sub(1,1)
   function package.searchpath(mod, path)
      mod = mod:gsub('%.', sep)
      for m in path:gmatch('[^;]+') do
         local nm = m:gsub('?', mod)
         local f = io.open(nm, 'r')
         if f then
            f:close()
            return nm
         end
     end
   end
end

-- load libTHNN
THNN.C = ffi.load(package.searchpath('libTHNN', package.cpath))

ffi.cdef(base_declarations)

-- expand macros, allow to use original lines from lib/THNN/generic/THNN.h
local preprocessed = string.gsub(generic_THNN_h, 'TH_API void THNN_%(([%a%d_]+)%)', 'void THNN_TYPE%1')

local replacements =
{
   {
      ['TYPE'] = 'Double',
      ['real'] = 'double',
      ['THTensor'] = 'THDoubleTensor',
      ['THIndexTensor'] = 'THLongTensor',
      ['THIntegerTensor'] = 'THIntTensor',
      ['THIndex_t'] = 'long',
      ['THInteger_t'] = 'int'
   },
   {
      ['TYPE'] = 'Float',
      ['real'] = 'float',
      ['THTensor'] = 'THFloatTensor',
      ['THIndexTensor'] = 'THLongTensor',
      ['THIntegerTensor'] = 'THIntTensor',
      ['THIndex_t'] = 'long',
      ['THInteger_t'] = 'int'
    }
}

for i=1,#replacements do
   local r = replacements[i]
   local s = preprocessed
   for k,v in pairs(r) do
      s = string.gsub(s, k, v)
   end
   ffi.cdef(s)
end

THNN.NULL = ffi.NULL or nil

function THNN.getState()
   return ffi.NULL or nil
end

function THNN.optionalTensor(t)
   return t and t:cdata() or THNN.NULL
end

local function extract_function_names(s)
   local t = {}
   for n in string.gmatch(s, 'TH_API void THNN_%(([%a%d_]+)%)') do
      t[#t+1] = n
   end
   return t
end

function THNN.bind(lib, base_names, type_name, state_getter)
   local ftable = {}
   local prefix = 'THNN_' .. type_name
   for i,n in ipairs(base_names) do
      -- use pcall since some libs might not support all functions (e.g. cunn)
      local ok,v = pcall(function() return lib[prefix .. n] end)
      if ok then
         ftable[n] = function(...) v(state_getter(), ...) end   -- implicitely add state
      else
         print('not found: ' .. prefix .. n .. v)
      end
   end
   return ftable
end

-- build function table
local function_names = extract_function_names(generic_THNN_h)

THNN.kernels = {}
THNN.kernels['torch.FloatTensor'] = THNN.bind(THNN.C, function_names, 'Float', THNN.getState)
THNN.kernels['torch.DoubleTensor'] = THNN.bind(THNN.C, function_names, 'Double', THNN.getState)

torch.getmetatable('torch.FloatTensor').THNN = THNN.kernels['torch.FloatTensor']
torch.getmetatable('torch.DoubleTensor').THNN = THNN.kernels['torch.DoubleTensor']

function THNN.runKernel(f, type, ...)
   local ftable = THNN.kernels[type]
   if not ftable then
      error('Unsupported tensor type: '..type)
   end
   local f = ftable[f]
   if not f then
      error(string.format("Function '%s' not found for tensor type '%s'.", f, type))
   end
   f(...)
end

return THNN
