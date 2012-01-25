#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/HardTanh.c"
#else

static int nn_(HardTanh_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  THTensor_(resizeAs)(output, input);

  TH_TENSOR_APPLY2(real, output, real, input, \
                   if(*input_data < -1)          \
                     *output_data = -1;          \
                   else if(*input_data <= 1)     \
                     *output_data = *input_data;    \
                   else                       \
                     *output_data = 1;)
  return 1;
}

static int nn_(HardTanh_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input, \
                   if(*input_data < -1 || *input_data > 1)               \
                     *gradInput_data = 0;                             \
                   else                                            \
                     *gradInput_data = *gradOutput_data;);
  return 1;
}

static const struct luaL_Reg nn_(HardTanh__) [] = {
  {"HardTanh_updateOutput", nn_(HardTanh_updateOutput)},
  {"HardTanh_updateGradInput", nn_(HardTanh_updateGradInput)},
  {NULL, NULL}
};

static void nn_(HardTanh_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(HardTanh__), "nn");
  lua_pop(L,1);
}

#endif
