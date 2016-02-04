------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
C-code header file for THNN package. Table 
'header' gets passed to an instance of the API
class during a call to API:c_init(). 

If passed as a second argument to API:__init__,
API instance will pass the header object to
API:c_init().


Authored: 2016-01-06
Modified: 2016-02-04
--]]

------------------------------------------------
--                                        THNN_h
------------------------------------------------
local header = {}

--------------------------------
--                    C Preamble
--------------------------------
--[[
* Preamble for C code; gets defined
first.
--]]
header['preamble'] =  
[[
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


--------------------------------
--                      Template
--------------------------------
--[[
* Pattern for re-expressing function signatures
prior to application of macros

* <tag>, <rtype> and <library> are special patterns
denoting references to config elements of the API
instance itself.

* The convention '$k' denotes retention of the k-th
capture group in the combined prefix-pattern
regular expression (see Forward Declaration).
--]]
header['template'] = '<rtype> <library>_TYPE$1'


--------------------------------
--                        Macros
--------------------------------
--[[
* Macro conventions:
  - Universal:    'old' = 'new'
  - Grouped:      <int> = {'old':'new', ...}
  - Alternatives: 'old' = {group_k:'new_k', ...}
--]]

header['macros'] =
{
  ['TYPE']            = {d='Double', f='Float'},
  ['real']            = {d='double', f='float'},
  ['THTensor']        = {d='THDoubleTensor', f='THFloatTensor'},
  ['THIndexTensor']   = 'THLongTensor',
  ['THIntegerTensor'] = 'THIntTensor',
  ['THIndex_t']       = 'long',
  ['THInteger_t']     = 'int',
}


--------------------------------
--           Forward Declaration
--------------------------------
--[[
* Forward declaration of C function

* Default Format:
  <tag> <rtype> <library>_<pattern>(...)
  |_______        _______|
       prefix

* Example: pattern := '%(([%a%d_]+)%)'
  
  <tag>    <rtype>   <library>_<pattern>
  TH_API    void    THNN_(Abs_updateOutput)
--]]

header['forward'] =
[[
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
      int p);
  TH_API void THNN_(MultiMarginCriterion_updateGradInput)(
      THNNState *state,
      THTensor *input,
      THTensor *target,
      THTensor *gradInput,
      bool sizeAverage,
      int p);
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
      THNNState *state,
      THTensor *input,
      THTensor *output,
      THTensor *weight,
      THTensor *bias,
      THTensor *finput,
      THTensor *fgradInput,
      int dT, int dW, int dH,
      int pT, int pW, int pH);
  TH_API void THNN_(VolumetricFullConvolution_updateGradInput)(
      THNNState *state,
      THTensor *input,
      THTensor *gradOutput,
      THTensor *gradInput,
      THTensor *weight,
      THTensor *finput,
      THTensor *fgradInput,
      int dT, int dW, int dH,
      int pT, int pW, int pH);
  TH_API void THNN_(VolumetricFullConvolution_accGradParameters)(
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
]]

return header 