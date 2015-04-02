#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialConvolutionMM.c"
#else

/* note: due to write issues, this one cannot be parallelized as well as unfolded_copy */
static void nn_(unfolded_acc)(THTensor *finput, THTensor *input,
                               int kW, int kH,
                               int dW, int dH,
                               int padding,
                               int nInputPlane,
                               int inputWidth, int inputHeight,
                               int outputWidth, int outputHeight)
{
  int nip;
  real *input_data = THTensor_(data)(input);
  real *finput_data = THTensor_(data)(finput);

#pragma omp parallel for private(nip)
  for(nip = 0; nip < nInputPlane; nip++)
  {
    int kw, kh, y, x, ix, iy;
    for(kh = 0; kh < kH; kh++)
    {
      for(kw = 0; kw < kW; kw++)
      {
        real *src = finput_data + nip*(kH*kW*outputHeight*outputWidth) + kh*(kW*outputHeight*outputWidth) + kw*(outputHeight*outputWidth);
        real *dst = input_data + nip*(inputHeight*inputWidth);
        if (padding > 0) {
          int lpad,rpad;
          for(y = 0; y < outputHeight; y++) {
            iy = y*dH - padding + kh;
            if (iy < 0 || iy >= inputHeight) {
            } else {
              if (dW==1){
                 ix = 0 - padding + kw;
                 lpad = fmaxf(0,padding-kw);
                 rpad = fmaxf(0,padding-(kW-kw-1));
                 THVector_(add)(dst+iy*inputWidth+ix+lpad, src+y*outputWidth+lpad, 1, outputWidth - lpad - rpad); /* note: THVector_add could handle 1 value better */
              }
              else{
                for (x=0; x<outputWidth; x++){
                   ix = x*dW - padding + kw;
                   if (ix < 0 || ix >= inputWidth){
                   }else
                     THVector_(add)(dst+iy*inputWidth+ix, src+y*outputWidth+x, 1, 1);
                }
              }
            }
          }
        } else {
          for(y = 0; y < outputHeight; y++) {
            iy = y*dH + kh;
            ix = 0 + kw;
            if (dW == 1 )
               THVector_(add)(dst+iy*inputWidth+ix, src+y*outputWidth, 1, outputWidth); /* note: THVector_add could handle 1 value better */
            else{
              for(x = 0; x < outputWidth; x++)
                THVector_(add)(dst+iy*inputWidth+ix+x*dW, src+y*outputWidth+x, 1, 1);
            }
          }
        }
      }
    }
  }
}

static void nn_(unfolded_copy)(THTensor *finput, THTensor *input,
                               int kW, int kH,
                               int dW, int dH,
                               int padding,
                               int nInputPlane,
                               int inputWidth, int inputHeight,
                               int outputWidth, int outputHeight)
{
  long k;
  real *input_data = THTensor_(data)(input);
  real *finput_data = THTensor_(data)(finput);

#pragma omp parallel for private(k)
  for(k = 0; k < nInputPlane*kH*kW; k++) {
    int nip = k / (kH*kW);
    int rest = k % (kH*kW);
    int kh = rest / kW;
    int kw = rest % kW;
    int x,y,ix,iy;
    real *dst = finput_data + nip*(kH*kW*outputHeight*outputWidth) + kh*(kW*outputHeight*outputWidth) + kw*(outputHeight*outputWidth);
    real *src = input_data + nip*(inputHeight*inputWidth);
    if (padding > 0) {
      int lpad,rpad;
      for(y = 0; y < outputHeight; y++) {
        iy = y*dH - padding + kh;
        if (iy < 0 || iy >= inputHeight) {
          memset(dst+y*outputWidth, 0, sizeof(real)*outputWidth);
        } else {
          if (dW==1){
             ix = 0 - padding + kw;
             lpad = fmaxf(0,padding-kw);
             rpad = fmaxf(0,padding-(kW-kw-1));
             if (lpad > 0) memset(dst+y*outputWidth, 0, sizeof(real)*lpad);
             memcpy(dst+y*outputWidth+lpad, src+iy*inputWidth+ix+lpad, sizeof(real)*(outputWidth-rpad-lpad));
             if (rpad > 0) memset(dst+y*outputWidth + outputWidth - rpad, 0, sizeof(real)*rpad);
          }
          else{
            for (x=0; x<outputWidth; x++){
               ix = x*dW - padding + kw;
               if (ix < 0 || ix >= inputWidth)
                 memset(dst+y*outputWidth+x, 0, sizeof(real)*1);
               else
                 memcpy(dst+y*outputWidth+x, src+iy*inputWidth+ix, sizeof(real)*(1));
            }
          }
        }
      }
    } else {
      for(y = 0; y < outputHeight; y++) {
        iy = y*dH + kh;
        ix = 0 + kw;
        if (dW == 1)
           memcpy(dst+y*outputWidth, src+iy*inputWidth+ix, sizeof(real)*outputWidth);
        else{
          for (x=0; x<outputWidth; x++)
             memcpy(dst+y*outputWidth+x, src+iy*inputWidth+ix+x*dW, sizeof(real)*(1));
         }
      }
    }
  }
}

static void nn_(SpatialConvolutionMM_updateOutput_frame)(THTensor *input, THTensor *output, THTensor *weight, THTensor *bias, THTensor *finput,
                                                         int kW, int kH, int dW, int dH, int padding,
                                                         long nInputPlane, long inputWidth, long inputHeight,
                                                         long nOutputPlane, long outputWidth, long outputHeight)
{
  long i;
  THTensor *output2d;

  nn_(unfolded_copy)(finput, input, kW, kH, dW, dH, padding, nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight);

  output2d = THTensor_(newWithStorage2d)(output->storage, output->storageOffset,
                                         nOutputPlane, -1,
                                         outputHeight*outputWidth, -1);

  for(i = 0; i < nOutputPlane; i++)
    THVector_(fill)(output->storage->data+output->storageOffset+output->stride[0]*i, THTensor_(get1d)(bias, i), outputHeight*outputWidth);

  THTensor_(addmm)(output2d, 1, output2d, 1, weight, finput);

  THTensor_(free)(output2d);
}

static int nn_(SpatialConvolutionMM_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int padding = luaT_getfieldcheckint(L, 1, "padding");

  THTensor *finput = luaT_getfieldcheckudata(L, 1, "finput", torch_Tensor);
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  int dimf = 0;
  int dimw = 2;
  int dimh = 1;

  long nInputPlane;
  long inputWidth;
  long inputHeight;
  long nOutputPlane;
  long outputWidth;
  long outputHeight;

  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D(batch mode) tensor expected");


  if (input->nDimension == 4) {
    dimf++;
    dimw++;
    dimh++;
  }

  nInputPlane = input->size[dimf];
  inputWidth   = input->size[dimw];
  inputHeight  = input->size[dimh];
  nOutputPlane = weight->size[0];
  outputWidth  = (inputWidth + 2*padding - kW) / dW + 1;
  outputHeight = (inputHeight + 2*padding - kH) / dH + 1;

  if(input->nDimension == 3)
  {
    THTensor_(resize2d)(finput, kW*kH*nInputPlane, outputHeight*outputWidth);
    THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);

    nn_(SpatialConvolutionMM_updateOutput_frame)(input, output, weight, bias, finput,
                                                 kW, kH, dW, dH, padding,
                                                 nInputPlane, inputWidth, inputHeight,
                                                 nOutputPlane, outputWidth, outputHeight);
  }
  else
  {
    long T = input->size[0];
    long t;

    THTensor_(resize3d)(finput, T, kW*kH*nInputPlane, outputHeight*outputWidth);
    THTensor_(resize4d)(output, T, nOutputPlane, outputHeight, outputWidth);

    THStorage_(clearFlag)(input->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(clearFlag)(output->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(clearFlag)(finput->storage, TH_STORAGE_REFCOUNTED);

#pragma omp parallel for private(t)
    for(t = 0; t < T; t++)
    {
      THTensor *input_t = THTensor_(newSelect)(input, 0, t);
      THTensor *output_t = THTensor_(newSelect)(output, 0, t);
      THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);

      nn_(SpatialConvolutionMM_updateOutput_frame)(input_t, output_t, weight, bias, finput_t,
                                                   kW, kH, dW, dH, padding,
                                                   nInputPlane, inputWidth, inputHeight,
                                                   nOutputPlane, outputWidth, outputHeight);

      THTensor_(free)(input_t);
      THTensor_(free)(output_t);
      THTensor_(free)(finput_t);
    }
    THStorage_(setFlag)(input->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(setFlag)(output->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(setFlag)(finput->storage, TH_STORAGE_REFCOUNTED);
  }

  return 1;
}


static void nn_(SpatialConvolutionMM_updateGradInput_frame)(THTensor *gradInput, THTensor *gradOutput, THTensor *weight, THTensor *fgradInput,
                                                            int kW, int kH, int dW, int dH, int padding)
{
  THTensor *gradOutput2d = THTensor_(newWithStorage2d)(gradOutput->storage, gradOutput->storageOffset,
                                                       gradOutput->size[0], -1,
                                                       gradOutput->size[1]*gradOutput->size[2], -1);
  THTensor_(addmm)(fgradInput, 0, fgradInput, 1, weight, gradOutput2d);
  THTensor_(free)(gradOutput2d);

  THTensor_(zero)(gradInput);

  nn_(unfolded_acc)(fgradInput, gradInput, kW, kH, dW, dH, padding, gradInput->size[0], gradInput->size[2], gradInput->size[1], gradOutput->size[2], gradOutput->size[1]);
}

static int nn_(SpatialConvolutionMM_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int padding = luaT_getfieldcheckint(L, 1, "padding");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  THTensor *finput = luaT_getfieldcheckudata(L, 1, "finput", torch_Tensor);
  THTensor *fgradInput = luaT_getfieldcheckudata(L, 1, "fgradInput", torch_Tensor);
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  THArgCheck( nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 1, "Number of output features is not equal to nOutputPlane" );

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(resizeAs)(fgradInput, finput);
  THTensor_(transpose)(weight, weight, 0, 1);

  if(input->nDimension == 3)
  {
    nn_(SpatialConvolutionMM_updateGradInput_frame)(gradInput, gradOutput, weight, fgradInput, kW, kH, dW, dH, padding);
  }
  else
  {
    long T = input->size[0];
    long t;

    THStorage_(clearFlag)(gradInput->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(clearFlag)(gradOutput->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(clearFlag)(fgradInput->storage, TH_STORAGE_REFCOUNTED);

#pragma omp parallel for private(t)
    for(t = 0; t < T; t++)
    {
      THTensor *gradInput_t = THTensor_(newSelect)(gradInput, 0, t);
      THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
      THTensor *fgradInput_t = THTensor_(newSelect)(fgradInput, 0, t);

      nn_(SpatialConvolutionMM_updateGradInput_frame)(gradInput_t, gradOutput_t, weight, fgradInput_t, kW, kH, dW, dH, padding);

      THTensor_(free)(gradInput_t);
      THTensor_(free)(gradOutput_t);
      THTensor_(free)(fgradInput_t);
    }

    THStorage_(setFlag)(gradInput->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(setFlag)(gradOutput->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(setFlag)(fgradInput->storage, TH_STORAGE_REFCOUNTED);
  }

  THTensor_(transpose)(weight, weight, 0, 1);

  return 1;
}

static void nn_(SpatialConvolutionMM_accGradParameters_frame)(THTensor *gradOutput, THTensor *gradWeight, THTensor *gradBias, THTensor *finput,
                                                              real scale)
{
  long i;
  THTensor *gradOutput2d = THTensor_(newWithStorage2d)(gradOutput->storage, gradOutput->storageOffset,
                                                       gradOutput->size[0], -1,
                                                       gradOutput->size[1]*gradOutput->size[2], -1);

  THTensor_(transpose)(finput, finput, 0, 1);
  THTensor_(addmm)(gradWeight, 1, gradWeight, scale, gradOutput2d, finput);
  THTensor_(transpose)(finput, finput, 0, 1);

  for(i = 0; i < gradBias->size[0]; i++)
  {
    long k;
    real sum = 0;
    real *data = gradOutput2d->storage->data + gradOutput2d->storageOffset + i*gradOutput2d->stride[0];
    for(k = 0; k < gradOutput2d->size[1]; k++)
      sum += data[k];
    (gradBias->storage->data + gradBias->storageOffset)[i] += scale*sum;
  }

  THTensor_(free)(gradOutput2d);
}

static int nn_(SpatialConvolutionMM_accGradParameters)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  real scale = luaL_optnumber(L, 4, 1);
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  THTensor *finput = luaT_getfieldcheckudata(L, 1, "finput", torch_Tensor);
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);

  THArgCheck( nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 1, "Number of output features is not equal to nOutputPlane" );

  if(input->nDimension == 3)
  {
    nn_(SpatialConvolutionMM_accGradParameters_frame)(gradOutput, gradWeight, gradBias, finput, scale);
  }
  else
  {
    long T = input->size[0];
    long t;

    for(t = 0; t < T; t++)
    {
      THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
      THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);

      nn_(SpatialConvolutionMM_accGradParameters_frame)(gradOutput_t, gradWeight, gradBias, finput_t, scale);

      THTensor_(free)(gradOutput_t);
      THTensor_(free)(finput_t);
    }
  }

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
