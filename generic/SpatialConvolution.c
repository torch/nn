#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialConvolution.c"
#else

static int nn_(SpatialConvolution_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

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
    /* add bias */
    long i;
    /*THTensor *outn = THTensor_(new)();*/
    real* bias_data = THTensor_(data)(bias);
    real* output_data = THTensor_(data)(output);
#pragma omp parallel for private(i)
    for (i=0; i<bias->size[0]; i++)
    {
      /*THTensor_(select)(outn,output,0,i);*/
      /*TH_TENSOR_APPLY(real,outn, *outn_data = bias_data[i];);*/
      real *ptr_output = output_data + i*outputWidth*outputHeight;
      long j;
      for(j = 0; j < outputWidth*outputHeight; j++)
      ptr_output[j] = bias_data[i];
    }
    /*THTensor_(free)(outn);*/
    
    /* do convolutions */
    THTensor_(conv2Dmv)(output, 1.0, 1.0, input, weight, dH, dW, "V","X");
  }
  else
  {
    THTensor_(resize4d)(output, input->size[0], nOutputPlane, outputHeight, outputWidth);

    real* bias_data = THTensor_(data)(bias);
    real* output_data = THTensor_(data)(output);

    long p;
#pragma omp parallel for private(p)
    for (p=0; p<input->size[0]; p++)
    {
      /* BIAS */
      long i;
      for (i=0; i<bias->size[0]; i++)
      {
        real *ptr_output = output_data + p*nOutputPlane*outputWidth*outputHeight + i*outputWidth*outputHeight;
        long j;
        for(j = 0; j < outputWidth*outputHeight; j++)
          ptr_output[j] = bias_data[i];
      }
    }

    /* do convolutions */
    THTensor_(conv2Dmm)(output, 1.0, 1.0, input, weight, dH, dW, "V","X");
  }
  return 1;
}


static int nn_(SpatialConvolution_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  THArgCheck( nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 1, "Number of output features is not equal to nOutputPlane" );

  /* gradient to input */
  THTensor *tweight = THTensor_(newTranspose)(weight,0,1);

  if (input->nDimension == 3)
  {
    THTensor_(conv2Dmv)(gradInput, 0.0, 1.0, gradOutput, tweight, dH, dW, "F","C");
  }
  else
  {
    THTensor_(conv2Dmm)(gradInput, 0.0, 1.0, gradOutput, tweight, dH, dW, "F","C");
  }
  THTensor_(free)(tweight);
  return 1;
}


static int nn_(SpatialConvolution_accGradParameters)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  real scale = luaL_optnumber(L, 4, 1);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);

  THArgCheck( nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 1, "Number of output features is not equal to nOutputPlane" );

  int dimw = 2;
  int dimh = 1;

  if (input->nDimension == 4)
  {
    dimw++;
    dimh++;
  }

  /* gradient to bias */
  real *gradBias_data = THTensor_(data)(gradBias);
  real *gradOutput_data = THTensor_(data)(gradOutput);
  long noutSlice = gradOutput->size[dimh]*gradOutput->size[dimw];
  /*THTensor* gradOutSlice = THTensor_(new)();*/

  if (input->nDimension == 3)
  {
    long k;
#pragma omp parallel for private(k)
    for(k = 0; k < nOutputPlane; k++)
    {
      /*THTensor_(select)(gradOutSlice, gradOutput, 0, k);*/
      real *ptr_gradOutput = gradOutput_data + k*noutSlice;
      long l;
      for(l = 0; l < noutSlice; l++)
        gradBias_data[k] += scale*ptr_gradOutput[l];
    }
    
    /* gradient to kernels */
    THTensor_(conv2DRevger)(gradWeight, 1.0, scale, input, gradOutput, dH, dW);
  }
  else
  {
    long k;
#pragma omp parallel for private(k)
    for(k = 0; k < nOutputPlane; k++)
    {
      long p;
      for(p = 0; p < input->size[0]; p++)
      { 
        /* BIAS */
        real *ptr_gradOutput = gradOutput_data + p*nOutputPlane*noutSlice + k*noutSlice;
        long l;
        for(l = 0; l < noutSlice; l++)
          gradBias_data[k] += scale*ptr_gradOutput[l];
      }
    }
    /* gradient to kernels */
    THTensor_(conv2DRevgerm)(gradWeight, 1.0, scale, input, gradOutput, dH, dW);
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
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SpatialConvolution__), "nn");
  lua_pop(L,1);
}

#endif
