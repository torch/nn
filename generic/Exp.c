#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Exp.c"
#else

static int nn_(Exp_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  
  THTensor_(resizeAs)(output, input);

  TH_TENSOR_APPLY2(real, output, real, input,         \
                   *output_data = exp(*input_data);)
    
  return 1;
}

static int nn_(Exp_updateGradInput)(lua_State *L)
{
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  THTensor_(resizeAs)(gradInput, output);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,     \
                   *gradInput_data = *gradOutput_data * *output_data;);
  return 1;
}

static const struct luaL_Reg nn_(Exp__) [] = {
  {"Exp_updateOutput", nn_(Exp_updateOutput)},
  {"Exp_updateGradInput", nn_(Exp_updateGradInput)},
  {NULL, NULL}
};

static void nn_(Exp_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(Exp__), "nn");
  lua_pop(L,1);
}

#endif
