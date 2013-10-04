#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SparseLinear.c"
#else

static int nn_(SparseLinear_updateOutput)(lua_State *L)
{
  long i;
  THTensor * input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor * weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor * bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor * output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  long dim = weight->size[0]; /* number of weights.. */

  THTensor_(copy)(output, bias);
  for(i = 0; i < input->size[1]; i++)
  {
    long offset = (long)(THTensor_(get2d)(input, 0, i))-1;
    
    if(offset >= 0 && offset < dim) /* make sure indices are in bounds.. */
    {
      real val = THTensor_(get2d)(input, 1, i);
      THBlas_(axpy)(output->size[0], 
                    val, 
                    THTensor_(data)(weight)+offset*weight->stride[0],
                    weight->stride[1], 
                    THTensor_(data)(output), 
                    output->stride[0]);
    }
    else
      luaL_error(L, "index out of bound");
  }
  return 1;
}

static int nn_(SparseLinear_accGradParameters)(lua_State *L)
{
  long i;
  THTensor * input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor * gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  real scale = luaL_optnumber(L, 4, 1);
  THTensor * weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor * gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);
  THTensor * gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
  THTensor * lastInput = luaT_getfieldcheckudata(L, 1, "lastInput", torch_Tensor);
  real weightDecay = luaT_getfieldchecknumber(L, 1, "weightDecay");
  long dim = gradWeight->size[0]; /* number of weights.. */

  for(i = 0; i < input->size[1]; i++)
  {
    long offset = (long)(THTensor_(get2d)(input, 0, i))-1;

    if(offset >= 0 && offset < dim) /* make sure indices are in bounds.. */
    {
      real val = scale*THTensor_(get2d)(input, 1, i);
      THBlas_(scal)(gradOutput->size[0],
                    0, 
                    THTensor_(data)(gradWeight)+offset*gradWeight->stride[0],
                    gradWeight->stride[1]); /* zero */

      THBlas_(axpy)(gradOutput->size[0], 
                    val, 
                    THTensor_(data)(gradOutput), 
                    gradOutput->stride[0], 
                    THTensor_(data)(gradWeight)+offset*gradWeight->stride[0], 
                    gradWeight->stride[1]);
    }
    else
      luaL_error(L, "index out of bound");
  }
  
  THTensor_(cadd)(gradBias, gradBias, 1, gradOutput); 
  
  if(weightDecay != 0)
    THTensor_(cadd)(gradWeight, gradWeight, weightDecay, weight);
  
  THTensor_(resizeAs)(lastInput, input);
  THTensor_(copy)(lastInput, input);
  
  return 0;
}

int nn_(SparseLinear_updateParameters)(lua_State *L)
{
  long i;
  real learningRate = luaL_checknumber(L, 2);
  THTensor * weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor * bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor * gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);
  THTensor * gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
  THTensor * lastInput = luaT_getfieldcheckudata(L, 1, "lastInput", torch_Tensor);
  
  long dim = weight->size[0]; /* number of weights.. */
  THTensor_(cadd)(bias, bias, -learningRate, gradBias);
  
  for(i = 0; i < lastInput->size[1]; i++) 
  {
    long offset = (long)(THTensor_(get2d)(lastInput, 0, i))-1;
    
    if(offset >= 0 && offset < dim) /* make sure indices are in bounds.. */
    {
      THBlas_(axpy)(bias->size[0], 
                    -learningRate, 
                    THTensor_(data)(gradWeight)+offset*gradWeight->stride[0], 
                    gradWeight->stride[1], 
                    THTensor_(data)(weight)+offset*weight->stride[0], 
                    weight->stride[1]);
    }
    else
      luaL_error(L, "index out of bound");
  }
  return 0;
}

static const struct luaL_Reg nn_(SparseLinear__) [] = {
  {"SparseLinear_updateOutput", nn_(SparseLinear_updateOutput)},
  {"SparseLinear_accGradParameters", nn_(SparseLinear_accGradParameters)},
  {"SparseLinear_updateParameters", nn_(SparseLinear_updateParameters)},
  {NULL, NULL}
};

void nn_(SparseLinear_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SparseLinear__), "nn");
  lua_pop(L,1);
}

#endif
