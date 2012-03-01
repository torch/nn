#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialConvolution.c"
#else

static void nn_(convolution_updateOutput_)(THTensor *input, THTensor *output, THTensor *weight, THTensor *bias, int dH, int dW)
{
  /* add bias */
  long i;
  THTensor *outn = THTensor_(new)();
  for (i=0; i<bias->size[0]; i++) {
    THTensor_(select)(outn,output,0,i);
    THTensor_(fill)(outn, THTensor_(get1d)(bias, i));
  }
  THTensor_(free)(outn);

  /* do convolutions */
  THTensor_(conv2Dmv)(output, 1.0, 1.0, input, weight, dH, dW, "V","X");
}

static int nn_(SpatialConvolution_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D(batch mode) tensor expected");

  int dimw = 2;
  int dimh = 1;
  if (input->nDimension == 4) {
    dimw++;
    dimh++;
  }

  long nOutputPlane = weight->size[0];
  long kW           = weight->size[3];
  long kH           = weight->size[2];
  long inputWidth   = input->size[dimw];
  long inputHeight  = input->size[dimh];
  long outputWidth  = (inputWidth - kW) / dW + 1;
  long outputHeight = (inputHeight - kH) / dH + 1;

  if (input->nDimension == 3)
  {
    THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);
/*     printf("\n*************\nstochastic\n"); */
/*     printf("no=%d\n",output->nDimension); */
/*     printf("no=%ld,%ld,%ld\n",nOutputPlane,outputHeight,outputWidth); */
/*     printf("ni=%d\n",input->nDimension); */
    nn_(convolution_updateOutput_)(input,output,weight,bias,dH,dW);
/*    printf("stochastic\n");*/
  }
  else
  {
    THTensor_(resize4d)(output, input->size[0], nOutputPlane, outputHeight, outputWidth);
    THTensor *outn = THTensor_(new)();
    THTensor *inpn = THTensor_(new)();
    long i;
    for (i=0; i<input->size[0]; i++)
    {
      THTensor_(select)(outn,output,0,i);
      THTensor_(select)(inpn,input,0,i);
      nn_(convolution_updateOutput_)(inpn,outn,weight,bias,dH,dW);
    }
    THTensor_(free)(outn);
    THTensor_(free)(inpn);
  }

/*   /\* add bias *\/ */
/*   long i; */
/*   THTensor *outn = THTensor_(new)(); */
/*   for (i=0; i<bias->size[0]; i++) { */
/*     THTensor_(select)(outn,output,0,i); */
/*     THTensor_(fill)(outn, THTensor_(get1d)(bias, i)); */
/*   } */
/*   THTensor_(free)(outn); */

/*   /\* do convolutions *\/ */
/*   THTensor_(conv2Dmv)(output, 1.0, 1.0, input, weight, dH, dW, "vx"); */

  return 1;
}


static int nn_(SpatialConvolution_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));  
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  THArgCheck( nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 1, "Number of output features is not equal to nOutputPlane" );

  /* gradient to input */
  THTensor *tweight = THTensor_(newTranspose)(weight,0,1);

  if(input->nDimension == 3)
  {
    THTensor_(conv2Dmv)(gradInput, 0.0, 1.0, gradOutput, tweight, dH, dW, "F", "C");
  }
  else
  {

    THTensor_(resizeAs)(gradInput,input);
    THTensor *outn = THTensor_(new)();
    THTensor *inpn = THTensor_(new)();
    long i;
    for (i=0; i<input->size[0]; i++)
    {
      THTensor_(select)(outn,gradOutput,0,i);
      THTensor_(select)(inpn,gradInput,0,i);
      THTensor_(conv2Dmv)(inpn, 0.0, 1.0, outn, tweight, dH, dW, "F", "C");
    }
    THTensor_(free)(outn);
    THTensor_(free)(inpn);
  }
  THTensor_(free)(tweight);

  return 1;
}

static void nn_(convolution_accGradParameters_)(THTensor *input, THTensor *gradOutput, THTensor *gradWeight, THTensor *gradBias, real scale, int dH, int dW)
{
  long k;

  /* gradient to bias */
  real *gradBias_data = THTensor_(data)(gradBias);
  THTensor* gradOutSlice = THTensor_(new)();
  for(k = 0; k < gradOutput->size[0]; k++)
  {
    THTensor_(select)(gradOutSlice, gradOutput, 0, k);
    gradBias_data[k] += scale*THTensor_(sumall)(gradOutSlice);
  }
  THTensor_(free)(gradOutSlice);

  /* gradient to kernels */
  THTensor_(conv2DRevger)(gradWeight, 1.0, scale, input, gradOutput, dH, dW);
}

static int nn_(SpatialConvolution_accGradParameters)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  real scale = luaL_optnumber(L, 4, 1);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_(Tensor_id));
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_(Tensor_id));
  
  THArgCheck( nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 1, "Number of output features is not equal to nOutputPlane" );

  if(input->nDimension == 3)
  {
    nn_(convolution_accGradParameters_)(input,gradOutput,gradWeight,gradBias,scale,dH,dW);
  }
  else
  {
    THTensor *outn = THTensor_(new)();
    THTensor *inpn = THTensor_(new)();
    long i;
    for (i=0; i<input->size[0]; i++)
    {
      THTensor_(select)(outn,gradOutput,0,i);
      THTensor_(select)(inpn,input,0,i);
      nn_(convolution_accGradParameters_)(inpn,outn,gradWeight,gradBias,scale,dH,dW);
    }
    THTensor_(free)(outn);
    THTensor_(free)(inpn);
  }

  return 0;
}

static const struct luaL_Reg nn_(SpatialConvolution__) [] = {
  {"SpatialConvolution_updateOutput", nn_(SpatialConvolution_updateOutput)},
  {"SpatialConvolution_updateGradInput", nn_(SpatialConvolution_updateGradInput)},
  {"SpatialConvolution_accGradParameters", nn_(SpatialConvolution_accGradParameters)},
  {NULL, NULL}
};

static void nn_(SpatialConvolution_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(SpatialConvolution__), "nn");
  lua_pop(L,1);
}

#endif
