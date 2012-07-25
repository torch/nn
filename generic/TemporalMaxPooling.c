#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/TemporalMaxPooling.c"
#else

static int nn_(TemporalMaxPooling_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  luaL_argcheck(L, input->nDimension == 2, 2, "2D tensor expected");
  luaL_argcheck(L, input->size[0] >= kW, 2, "input sequence smaller than kernel size");

  // sizes
  long niframe = input->size[0];
  long framesize = input->size[1];
  long noframe = (niframe - kW) / dW + 1;

  // get contiguous input
  input = THTensor_(newContiguous)(input);

  // resize output
  THTensor_(resize2d)(output, noframe, framesize);

  // indices will contain index locations for each output point
  THTensor_(resize2d)(indices, noframe, framesize);

  // get raw pointers
  real *input_data = THTensor_(data)(input);
  real *output_data = THTensor_(data)(output);
  real *indices_data = THTensor_(data)(indices);

  long t, x, y;
  for(t = 0; t < noframe; t++)
  {
    real *ip = input_data + t*framesize*dW;
    real *op = output_data + t*framesize;
    real *xp = indices_data + t*framesize;
#pragma omp parallel for private(y)
    for(y = 0; y < framesize; y++)
    {
      // compute local max:
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

      // set output to local max
      op[y] = maxval;
      xp[y] = (real)maxindex;
    }
  }

  // cleanup
  THTensor_(free)(input);

  return 1;
}

static int nn_(TemporalMaxPooling_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  // get contiguous gradOutput
  gradOutput = THTensor_(newContiguous)(gradOutput);

  // resize and zero
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  // sizes
  int noframe = gradOutput->size[0];
  long framesize = gradOutput->size[1];

  // get raw pointers
  real *gradInput_data = THTensor_(data)(gradInput);
  real *gradOutput_data = THTensor_(data)(gradOutput);
  real *indices_data = THTensor_(data)(indices);

  long t, y;
  for(t = 0; t < noframe; t++)
  {
    real *gip = gradInput_data + t*framesize*dW;
    real *gop = gradOutput_data + t*framesize;
    real *xp = indices_data + t*framesize;
#pragma omp parallel for private(y)
    for(y = 0; y < framesize; y++)
    {
      // compute local max:
      long maxindex = (long)xp[y];
      gip[maxindex*framesize+y] += gop[y];
    }
  }

  // cleanup
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
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(TemporalMaxPooling__), "nn");
  lua_pop(L,1);
}

#endif
