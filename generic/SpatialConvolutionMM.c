#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialConvolutionMM.c"
#else

static int nn_(SpatialConvolutionMM_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  long i;

  THTensor *finput = luaT_getfieldcheckudata(L, 1, "finput", torch_Tensor);
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D(batch mode) tensor expected");

  int dimw = 2;
  int dimh = 1;
  if (input->nDimension == 4) {
    THError("batch not yet supported");
  }

  long nInputPlane = input->size[0];
  long inputWidth   = input->size[dimw];
  long inputHeight  = input->size[dimh];
  long nOutputPlane = weight->size[0];
  long outputWidth  = (inputWidth - kW) / dW + 1;
  long outputHeight = (inputHeight - kH) / dH + 1;
  
  THTensor *unfoldedInput = THTensor_(new)();
  THTensor_(unfold)(unfoldedInput, input, 1, kH, dH);
  THTensor_(unfold)(unfoldedInput, unfoldedInput, 2, kW, dW);
  THTensor_(transpose)(unfoldedInput, unfoldedInput, 1, 3);
  THTensor_(transpose)(unfoldedInput, unfoldedInput, 2, 4);

  THTensor_(resize2d)(finput, kW*kH*nInputPlane, outputHeight*outputWidth);
  THTensor_(copy)(finput, unfoldedInput);

  THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);
  THTensor *output2d = THTensor_(newWithStorage2d)(output->storage, 0,
                                                   nOutputPlane, -1,
                                                   outputHeight*outputWidth, -1);

  for(i = 0; i < nOutputPlane; i++)
    THVector_(fill)(output->storage->data+output->storageOffset+output->stride[0]*i, THTensor_(get1d)(bias, i), outputHeight*outputWidth);

  THTensor_(addmm)(output2d, 1, output2d, 1, weight, finput);

  THTensor_(free)(output2d);
  THTensor_(free)(unfoldedInput);

  return 1;
}


static int nn_(SpatialConvolutionMM_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  THTensor *finput = luaT_getfieldcheckudata(L, 1, "finput", torch_Tensor);
  THTensor *fgradInput = luaT_getfieldcheckudata(L, 1, "fgradInput", torch_Tensor);
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  THArgCheck( nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 1, "Number of output features is not equal to nOutputPlane" );

  THTensor *gradOutput2d = THTensor_(newWithStorage2d)(gradOutput->storage, 0,
                                                       gradOutput->size[0], -1,
                                                       gradOutput->size[1]*gradOutput->size[2], -1);

  THTensor_(resizeAs)(fgradInput, finput);
  THTensor_(zero)(fgradInput);
  THTensor_(transpose)(weight, weight, 0, 1);
  THTensor_(addmm)(fgradInput, 1, fgradInput, 1, weight, gradOutput2d);
  THTensor_(transpose)(weight, weight, 0, 1);

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  THTensor *unfoldedGradInput = THTensor_(new)();
  THTensor_(unfold)(unfoldedGradInput, gradInput, 1, kH, dH);
  THTensor_(unfold)(unfoldedGradInput, unfoldedGradInput, 2, kW, dW);
  THTensor_(transpose)(unfoldedGradInput, unfoldedGradInput, 1, 3);
  THTensor_(transpose)(unfoldedGradInput, unfoldedGradInput, 2, 4);
      
  THTensor_(cadd)(unfoldedGradInput, unfoldedGradInput, 1, fgradInput);

  THTensor_(free)(unfoldedGradInput);
  THTensor_(free)(gradOutput2d);

  return 1;
}


static int nn_(SpatialConvolutionMM_accGradParameters)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  real scale = luaL_optnumber(L, 4, 1);
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  long i;

  THTensor *finput = luaT_getfieldcheckudata(L, 1, "finput", torch_Tensor);
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);

  THArgCheck( nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 1, "Number of output features is not equal to nOutputPlane" );

  THTensor *gradOutput2d = THTensor_(newWithStorage2d)(gradOutput->storage, 0,
                                                       gradOutput->size[0], -1,
                                                       gradOutput->size[1]*gradOutput->size[2], -1);

  THTensor_(transpose)(finput, finput, 0, 1);
  THTensor_(addmm)(gradWeight, 1, gradWeight, scale, gradOutput2d, finput);
  THTensor_(transpose)(finput, finput, 0, 1);

  THTensor *gradOutputPlane = THTensor_(new)();
  for(i = 0; i < gradBias->size[0]; i++)
  {
    THTensor_(select)(gradOutputPlane, gradOutput2d, 0, i);
    gradBias->storage->data[i] += scale*THTensor_(sumall)(gradOutputPlane);
  }

  THTensor_(free)(gradOutputPlane);
  THTensor_(free)(gradOutput2d);

  return 0;
}

static const struct luaL_Reg nn_(SpatialConvolutionMM__) [] = {
  {"SpatialConvolutionMM_updateOutput", nn_(SpatialConvolutionMM_updateOutput)},
  {"SpatialConvolutionMM_updateGradInput", nn_(SpatialConvolutionMM_updateGradInput)},
  {"SpatialConvolutionMM_accGradParameters", nn_(SpatialConvolutionMM_accGradParameters)},
  {NULL, NULL}
};

static void nn_(SpatialConvolutionMM_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SpatialConvolutionMM__), "nn");
  lua_pop(L,1);
}

#endif
