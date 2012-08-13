#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Min.c"
#else

static int nn_(Min_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int dimension = luaT_getfieldcheckint(L, 1, "dimension")-1;
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  THLongStorage *dim;
  long i;

  luaL_argcheck(L, dimension >= 0 && dimension < input->nDimension, 2, "dimension out of range");

  dim = THLongStorage_newWithSize(input->nDimension);
  for(i = 0; i < input->nDimension; i++)
    dim->data[i] = input->size[i];
  dim->data[dimension] = 1;
  THTensor_(resize)(output, dim, NULL);
  THTensor_(resize)(indices, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY3(real, output, real, input, real, indices, dimension,
                       long theIndex = 0;
                       real theMin = input_data[0];
                       for(i = 1; i < input_size; i++)
                       {
                         if(input_data[i*input_stride] < theMin)
                         {
                           theIndex = i;
                           theMin = input_data[i*input_stride];
                         }
                       }
                       *indices_data = theIndex+1;
                       *output_data = theMin;)

  THTensor_(select)(output, NULL, dimension, 0);

  return 1;
}

static int nn_(Min_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor);
  int dimension  = luaT_getfieldcheckint(L, 1, "dimension")-1;
  THTensor *gradInput  = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  THTensor *gradOutputPlusOneDim;
  THLongStorage *dim, *str;
  int i, j;

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  dim = THLongStorage_newWithSize(gradOutput->nDimension+1);
  str = THLongStorage_newWithSize(gradOutput->nDimension+1);
  for(i = 0, j =  0; j < gradOutput->nDimension+1; j++)
  {
    if(j == dimension)
    {
      dim->data[j] = input->size[dimension];
      str->data[j] = 0;
      continue;
    }

    dim->data[j] = gradOutput->size[i];
    str->data[j] = gradOutput->stride[i];
    i++;
  }

  gradOutputPlusOneDim = THTensor_(newWithStorage)(gradOutput->storage, gradOutput->storageOffset, dim, str);
  THLongStorage_free(dim);
  THLongStorage_free(str);

  TH_TENSOR_DIM_APPLY3(real, gradInput, real, gradOutputPlusOneDim, real, indices, dimension,
                       gradInput_data[ ((long)(*indices_data)-1)*gradInput_stride ] = *gradOutputPlusOneDim_data;)

  THTensor_(free)(gradOutputPlusOneDim);

  return 1;
}

static const struct luaL_Reg nn_(Min__) [] = {
  {"Min_updateOutput", nn_(Min_updateOutput)},
  {"Min_updateGradInput", nn_(Min_updateGradInput)},
  {NULL, NULL}
};

static void nn_(Min_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(Min__), "nn");
  lua_pop(L,1);
}

#endif
