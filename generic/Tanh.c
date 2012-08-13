#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Tanh.c"
#else

static int nn_(Tanh_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  THTensor_(resizeAs)(output, input);

  if (input->nDimension == 1 || !THTensor_(isContiguous)(input) || !THTensor_(isContiguous)(output))
  {
    TH_TENSOR_APPLY2(real, output, real, input,   \
         *output_data = tanh(*input_data););
  }
  else
  {
    real* output_data = THTensor_(data)(output);
    real* input_data  = THTensor_(data)(input);
    long k;

#pragma omp parallel for private(k)
    for (k = 0; k < input->size[0]; k++)
    {
      real* ptr_output = output_data + k*input->stride[0];
      real* ptr_input  = input_data  + k*input->stride[0];
      long i;
      for (i = 0; i < input->stride[0]; i++)
  ptr_output[i] = tanh(ptr_input[i]);
    }
  }
  return 1;
}

static int nn_(Tanh_updateGradInput)(lua_State *L)
{
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  THTensor_(resizeAs)(gradInput, output);

  if (output->nDimension == 1 || 
      !THTensor_(isContiguous)(output) || 
      !THTensor_(isContiguous)(gradOutput) ||
      !THTensor_(isContiguous)(gradInput))
  {
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,  \
         real z = *output_data;            \
         *gradInput_data = *gradOutput_data * (1. - z*z););
  }
  else
  {
    real* gradOutput_data = THTensor_(data)(gradOutput);
    real* gradInput_data  = THTensor_(data)(gradInput);
    real* output_data     = THTensor_(data)(output);
    long k;

#pragma omp parallel for private(k)
    for (k = 0; k < output->size[0]; k++)
    {
      real* ptr_gradOutput = gradOutput_data + k*output->stride[0];
      real* ptr_gradInput  = gradInput_data  + k*output->stride[0];
      real* ptr_output     = output_data     + k*output->stride[0];
      long i;
      for (i = 0; i < output->stride[0]; i++)
      {
  real z = ptr_output[i];
  ptr_gradInput[i] = ptr_gradOutput[i] * (1. - z*z);
      }
    }
  }
  return 1;
}

static const struct luaL_Reg nn_(Tanh__) [] = {
  {"Tanh_updateOutput", nn_(Tanh_updateOutput)},
  {"Tanh_updateGradInput", nn_(Tanh_updateGradInput)},
  {NULL, NULL}
};

static void nn_(Tanh_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(Tanh__), "nn");
  lua_pop(L,1);

}

#endif
