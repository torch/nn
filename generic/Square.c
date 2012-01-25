#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Square.c"
#else

static int nn_(Square_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));
  
  THTensor_(resizeAs)(output, input);

  TH_TENSOR_APPLY2(real, output, real, input,	\
		   *output_data = *input_data * *input_data;);

  return 1;
}

static int nn_(Square_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  THTensor_(resizeAs)(gradInput, input);

  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input, \
		   *gradInput_data = 2.0 * (*gradOutput_data) * (*input_data););

  return 1;
}

static const struct luaL_Reg nn_(Square__) [] = {
  {"Square_updateOutput", nn_(Square_updateOutput)},
  {"Square_updateGradInput", nn_(Square_updateGradInput)},
  {NULL, NULL}
};

static void nn_(Square_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(Square__), "nn");
  lua_pop(L,1);
}

#endif
