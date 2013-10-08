#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/TemporalMaxPooling.c"
#else

static int nn_(TemporalMaxPooling_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  long niframe;
  long framesize;
  long noframe;

  real *input_data;
  real *output_data;
  real *indices_data;

  long t, x, y;

  luaL_argcheck(L, input->nDimension == 2, 2, "2D tensor expected");
  luaL_argcheck(L, input->size[0] >= kW, 2, "input sequence smaller than kernel size");

  /* sizes */
  niframe = input->size[0];
  framesize = input->size[1];
  noframe = (niframe - kW) / dW + 1;

  /* get contiguous input */
  input = THTensor_(newContiguous)(input);

  /* resize output */
  THTensor_(resize2d)(output, noframe, framesize);

  /* indices will contain index locations for each output point */
  THTensor_(resize2d)(indices, noframe, framesize);

  /* get raw pointers */
  input_data = THTensor_(data)(input);
  output_data = THTensor_(data)(output);
  indices_data = THTensor_(data)(indices);

  for(t = 0; t < noframe; t++)
  {
    real *ip = input_data + t*framesize*dW;
    real *op = output_data + t*framesize;
    real *xp = indices_data + t*framesize;
#pragma omp parallel for private(y)
    for(y = 0; y < framesize; y++)
    {
      /* compute local max: */
      long maxindex = -1;
      real maxval = -THInf;
      for(x = 0; x < kW; x++)
      {
        real val = ip[x*framesize+y];
        if (val > maxval)
        {
          maxval = val;
          maxindex = x;
        }
      }

      /* set output to local max */
      op[y] = maxval;
      xp[y] = (real)maxindex;
    }
  }

  /* cleanup */
  THTensor_(free)(input);

  return 1;
}

static int nn_(TemporalMaxPooling_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  int noframe;
  long framesize;

  real *gradInput_data;
  real *gradOutput_data;
  real *indices_data;

  long t, y;

  /* get contiguous gradOutput */
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* resize and zero */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  /* sizes */
  noframe = gradOutput->size[0];
  framesize = gradOutput->size[1];

  /* get raw pointers */
  gradInput_data = THTensor_(data)(gradInput);
  gradOutput_data = THTensor_(data)(gradOutput);
  indices_data = THTensor_(data)(indices);

  for(t = 0; t < noframe; t++)
  {
    real *gip = gradInput_data + t*framesize*dW;
    real *gop = gradOutput_data + t*framesize;
    real *xp = indices_data + t*framesize;
#pragma omp parallel for private(y)
    for(y = 0; y < framesize; y++)
    {
      /* compute local max: */
      long maxindex = (long)xp[y];
      gip[maxindex*framesize+y] += gop[y];
    }
  }

  /* cleanup */
  THTensor_(free)(gradOutput);

  return 1;
}

static const struct luaL_Reg nn_(TemporalMaxPooling__) [] = {
  {"TemporalMaxPooling_updateOutput", nn_(TemporalMaxPooling_updateOutput)},
  {"TemporalMaxPooling_updateGradInput", nn_(TemporalMaxPooling_updateGradInput)},
  {NULL, NULL}
};

static void nn_(TemporalMaxPooling_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(TemporalMaxPooling__), "nn");
  lua_pop(L,1);
}

#endif
