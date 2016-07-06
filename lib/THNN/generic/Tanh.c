#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Tanh.c"
#else

void THNN_(Tanh_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output)
{
  input = THTensor_(newContiguous)(input);
  THTensor_(resizeAs)(output, input);

  real * in  = THTensor_(data)(input);
  real * out = THTensor_(data)(output);

#ifdef TH_REAL_IS_FLOAT
#pragma simd
  for (int j = 0; j < THTensor_(nElement)(input); j++) {
    out[j] = tanhf(in[j]);
  }
#else
#pragma simd
  for (int j = 0; j < THTensor_(nElement)(input); j++) {
    out[j] = tanh(in[j]);
  }
#endif
  THTensor_(free)(input);
}

void THNN_(Tanh_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output)
{
  THTensor_(resizeAs)(gradInput, output);

  if (output->nDimension == 1 || 
      !THTensor_(isContiguous)(output) || 
      !THTensor_(isContiguous)(gradOutput) ||
      !THTensor_(isContiguous)(gradInput))
  {
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,
      real z = *output_data;            \
      *gradInput_data = *gradOutput_data * (1. - z*z);
    );
  }
  else
  {
    real* ptr_gradOutput = THTensor_(data)(gradOutput);
    real* ptr_gradInput  = THTensor_(data)(gradInput);
    real* ptr_output     = THTensor_(data)(output);
    long i;

#pragma omp parallel for private(i)
    for (i = 0; i < THTensor_(nElement)(gradInput); i++)
    {
      real z = ptr_output[i];
      ptr_gradInput[i] = ptr_gradOutput[i] * (1. - z*z);
    }
  }
}

#endif
