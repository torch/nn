#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Tanh.c"
#else

static int nn_(Tanh_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int inPlace = luaT_getfieldcheckboolean(L, 1, "inplace");
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  if (inPlace) {
    TH_TENSOR_APPLY(real, input,                        \
                    *input_data = tanh(*input_data););
    THTensor_(set)(output, input);
  } else {
    THTensor_(resizeAs)(output, input);
    if (input->nDimension == 1 || !THTensor_(isContiguous)(input) || !THTensor_(isContiguous)(output)) {
      TH_TENSOR_APPLY2(real, output, real, input,       \
                       *output_data = tanh(*input_data););
    } else {
      real* ptr_output = THTensor_(data)(output);
      real* ptr_input  = THTensor_(data)(input);
      long i;
#pragma omp parallel for private(i)
      for(i = 0; i < THTensor_(nElement)(input); i++) {
        ptr_output[i] = tanh(ptr_input[i]);
      }
    }
  }
  return 1;
}

static int nn_(Tanh_updateGradInput)(lua_State *L)
{
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  int inPlace = luaT_getfieldcheckboolean(L, 1, "inplace");
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  if (inPlace) {
    TH_TENSOR_APPLY2(real, gradOutput, real, output,                    \
                     real z = *output_data;                             \
                     *gradOutput_data = *gradOutput_data * (1. - z*z););
    THTensor_(set)(gradInput, gradOutput);
  } else {
    THTensor_(resizeAs)(gradInput, output);
    if (output->nDimension == 1 ||
        !THTensor_(isContiguous)(output) ||
        !THTensor_(isContiguous)(gradOutput) ||
        !THTensor_(isContiguous)(gradInput)) {
      TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output, \
                       real z = *output_data;                           \
                       *gradInput_data = *gradOutput_data * (1. - z*z););
    } else {
      real* ptr_gradOutput = THTensor_(data)(gradOutput);
      real* ptr_gradInput  = THTensor_(data)(gradInput);
      real* ptr_output     = THTensor_(data)(output);
      long i;

#pragma omp parallel for private(i)
      for(i = 0; i < THTensor_(nElement)(gradInput); i++) {
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
