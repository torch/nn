#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Sigmoid.c"
#else

void THNN_(Sigmoid_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output)
{
  THTensor_(resizeAs)(output, input);
  real * in  = THTensor_(data)(input);
  real * out = THTensor_(data)(output);
  const int iStride =  input->stride[0];
  const int oStride = output->stride[0];
#ifdef TH_REAL_IS_FLOAT
  for (int i = 0; i < output->size[0]; i++) {
#pragma simd
    for (int j = 0; j < output->stride[0]; j++) {
      out[oStride*i+j] = 1.0f/(1.0f + expf(-in[iStride*i+j]));
    }
  }
#else
  for (int i = 0; i < output->size[0]; i++) {
#pragma simd
    for (int j = 0; j < output->stride[0]; j++) {
      out[oStride*i+j] = 1.0/(1.0 + exp(-in[iStride*i+j]));
    }
  }
#endif

}

void THNN_(Sigmoid_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output)
{
  THTensor_(resizeAs)(gradInput, output);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,
    real z = *output_data;
    *gradInput_data = *gradOutput_data * (1. - z) * z;
  );
}

#endif
