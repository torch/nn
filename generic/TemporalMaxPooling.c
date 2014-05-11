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

  long t, y;
  
  int dimS = 0; // sequence dimension
  int dimF = 1; // feature dimension

  luaL_argcheck(L, input->nDimension == 2 || input->nDimension == 3, 2, "2D or 3D(batch mode) tensor expected");
  
  if (input->nDimension == 3) 
  {
    dimS = 1;
    dimF = 2;
  }
  luaL_argcheck(L, input->size[dimS] >= kW, 2, "input sequence smaller than kernel size");
  
  /* sizes */
  niframe = input->size[dimS];
  framesize = input->size[dimF];
  noframe = (niframe - kW) / dW + 1;
   
  /* get contiguous input */
  input = THTensor_(newContiguous)(input);

  if (input->nDimension == 2)
  {
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
        long x;
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
  }
  else
  {
    /* number of batch frames */
    long nbframe = input->size[0];
    long i;
    
    /* resize output */
    THTensor_(resize3d)(output, nbframe, noframe, framesize);

    /* indices will contain index locations for each output point */
    THTensor_(resize3d)(indices, nbframe, noframe, framesize);
    
    /* get raw pointers */
    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THTensor_(data)(indices);
    
    for(i = 0; i < nbframe; i++)
    {
      real *inputSample_data = input_data + i*niframe*framesize;
      real *outputSample_data = output_data + i*noframe*framesize;
      real *indicesSample_data = indices_data + i*noframe*framesize;
      
      for(t = 0; t < noframe; t++)
      {
        real *ip = inputSample_data + t*framesize*dW;
        real *op = outputSample_data + t*framesize;
        real *xp = indicesSample_data + t*framesize;
        
#pragma omp parallel for private(y)
        for(y = 0; y < framesize; y++)
        {
          /* compute local max: */
          long maxindex = -1;
          real maxval = -THInf;
          long x;
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

  long niframe;
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

  int dimS = 0; // sequence dimension
  int dimF = 1; // feature dimension
  
  if (input->nDimension == 3) 
  {
    dimS = 1;
    dimF = 2;
  }
  /* sizes */
  niframe = input->size[dimS];
  noframe = gradOutput->size[dimS];
  framesize = gradOutput->size[dimF];

  /* get raw pointers */
  gradInput_data = THTensor_(data)(gradInput);
  gradOutput_data = THTensor_(data)(gradOutput);
  indices_data = THTensor_(data)(indices);

  if (input->nDimension == 2)
  {
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
  }
  else
  {
    /* number of batch frames */
    long nbframe = input->size[0];
    long i;
      
    for(i = 0; i < nbframe; i++)
    {
      real *gradInputSample_data = gradInput_data + i*niframe*framesize;
      real *gradOutputSample_data = gradOutput_data + i*noframe*framesize;
      real *indicesSample_data = indices_data + i*noframe*framesize;
      
      for(t = 0; t < noframe; t++)
      {
        real *gip = gradInputSample_data + t*framesize*dW;
        real *gop = gradOutputSample_data + t*framesize;
        real *xp = indicesSample_data + t*framesize;
#pragma omp parallel for private(y)
        for(y = 0; y < framesize; y++)
        {
          /* compute local max: */
          long maxindex = (long)xp[y];
          gip[maxindex*framesize+y] += gop[y];
        }
      }
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
