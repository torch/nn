#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Linear.c"
#else

static int nn_(Linear_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);

  THTensor *ones = luaT_getfieldcheckudata(L, 1, "_ones", torch_Tensor);
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  luaL_argcheck(L, input->nDimension == 1 || input->nDimension == 2, 2, "1D or 2D tensor expected");

  if (input->nDimension == 1) 
  {
    luaL_argcheck(L, input->size[0] == weight->size[1], 2, "invalid number of input units (input:size(1))");
    
    THTensor_(resize1d)(output, bias->size[0]);
    THTensor_(copy)(output, bias);
    THTensor_(addmv)(output, 1, output, 1, weight, input);
  }
  else if ( input->nDimension == 2 ) 
  {
    long nframe = input->size[0];
    long nunit = bias->size[0];
    
    luaL_argcheck(L, input->size[1] == weight->size[1], 2, "invalid number of input units (input:size(2))");

    THTensor_(resize2d)(output, nframe, nunit);
    if (ones->size[0] != nframe)
    {
      THTensor_(resize1d)(ones, nframe);
      THTensor_(fill)(ones, 1);
    }
      
    THTensor_(zero)(output);
    THTensor_(addr)(output, 1, output, 1, ones, bias);
    
    THTensor_(transpose)(weight, NULL, 0, 1);
    THTensor_(addmm)(output, 1, output, 1, input, weight);
    THTensor_(transpose)(weight, NULL, 0, 1);
  }

  return 1;
}

static int nn_(Linear_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  
  long nElement = THTensor_(nElement)(gradInput);
  
  luaL_argcheck(L, input->nDimension == 1 || input->nDimension == 2, 2, "1D or 2D tensor expected");
  luaL_argcheck(L, gradOutput->nDimension == input->nDimension, 2, "input and gradOutput should have same number of dimensions");
  
  THTensor_(resizeAs)(gradInput, input);
  if (THTensor_(nElement)(gradInput) != nElement)
    THTensor_(zero)(gradInput);
  
  if (input->nDimension == 1) 
  {
    luaL_argcheck(L, input->size[0] == weight->size[1], 2, "invalid number of input units (input:size(1))");
    luaL_argcheck(L, gradOutput->size[0] == weight->size[0], 2, "invalid number of output units (gradOutput:size(1))");
    
    THTensor_(transpose)(weight, NULL, 0, 1);
    THTensor_(addmv)(gradInput, 0, gradInput, 1, weight, gradOutput);
    THTensor_(transpose)(weight, NULL, 0, 1);
  }
  else
  {
    luaL_argcheck(L, input->size[1] == weight->size[1], 2, "invalid number of input units (input:size(2))");
    luaL_argcheck(L, gradOutput->size[1] == weight->size[0], 2, "invalid number of output units (gradOutput:size(2))");
    
    THTensor_(addmm)(gradInput, 0, gradInput, 1, gradOutput, weight);
  }
  
  return 1;
}

static int nn_(Linear_accGradParameters)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  real scale = luaL_optnumber(L, 4, 1);

  THTensor *ones = luaT_getfieldcheckudata(L, 1, "_ones", torch_Tensor);
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);
  
  luaL_argcheck(L, input->nDimension == 1 || input->nDimension == 2, 2, "1D or 2D tensor expected");
  luaL_argcheck(L, gradOutput->nDimension == input->nDimension, 2, "input and gradOutput should have same number of dimensions");
  
  if (input->nDimension == 1) 
  {
    luaL_argcheck(L, input->size[0] == gradWeight->size[1], 2, "invalid number of input units (input:size(1))");
    luaL_argcheck(L, gradOutput->size[0] == gradWeight->size[0], 2, "invalid number of output units (gradOutput:size(1))");
    
    THTensor_(addr)(gradWeight, 1, gradWeight, scale, gradOutput, input);
    THTensor_(cadd)(gradBias, gradBias, scale, gradOutput);
  }
  else 
  {
    long nframe = input->size[0];
    
    luaL_argcheck(L, input->size[1] == gradWeight->size[1], 2, "invalid number of input units (input:size(2))");
    luaL_argcheck(L, gradOutput->size[1] == gradWeight->size[0], 2, "invalid number of output units (gradOutput:size(2))");

    if (ones->size[0] != nframe)
    {
      THTensor_(resize1d)(ones, nframe);
      THTensor_(fill)(ones, 1);
    }
    THTensor_(transpose)(gradOutput, NULL, 0, 1);
    THTensor_(addmm)(gradWeight, 1, gradWeight, scale, gradOutput, input);
    THTensor_(addmv)(gradBias, 1, gradBias, scale, gradOutput, ones);
    THTensor_(transpose)(gradOutput, NULL, 0, 1);
  }

  return 0;
}

static const struct luaL_Reg nn_(Linear__) [] = {
  {"Linear_updateOutput", nn_(Linear_updateOutput)},
  {"Linear_updateGradInput", nn_(Linear_updateGradInput)},
  {"Linear_accGradParameters", nn_(Linear_accGradParameters)},
  {NULL, NULL}
};

static void nn_(Linear_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(Linear__), "nn");
  lua_pop(L,1);
}

#endif
