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


Authored: 2016-01-06 (jwilson)
Modified: 2016-02-03
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
denoting references to config elements of an API
instance

* The convention '$k' denotes retention of the k-th
capture group in the combined prefix-pattern
regular expression (see Forward Declaration).
--]]
header['template'] = '<rtype> <library>_<tensor>$1'


--------------------------------
--                        Macros
--------------------------------
--[[
* Macro convention: {new = 'old'}
* Subtables processed seperately
--]]

header['macros'] =
{
  {Double='<tensor>', double='real', THDoubleTensor='THTensor', THLongTensor='THIndexTensor'},
  {Float ='<tensor>', float ='real', THFloatTensor ='THTensor', THLongTensor='THIndexTensor'},
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
]]

return header