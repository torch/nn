------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Header file for THNN package.

Authored: 2016-01-06 (jwilson)
Modified: 2016-01-06
--]]


------------------------------------------------
--                                        THNN_h
------------------------------------------------
local header =
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