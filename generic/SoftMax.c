#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SoftMax.c"
#else

static int nn_(SoftMax_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);  
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  real *input_data, *output_data;
  long nframe = 0, dim = 0;
  long t, d;

  if(input->nDimension == 1)
  {
    nframe = 1;
    dim = input->size[0];
  }
  else if(input->nDimension == 2)
  {
    nframe = input->size[0];
    dim = input->size[1];
  }
  else
    THArgCheck(0, 2, "vector or matrix expected");

  input = THTensor_(newContiguous)(input);
  THTensor_(resizeAs)(output, input);

  input_data = THTensor_(data)(input);
  output_data = THTensor_(data)(output);
  for(t = 0; t < nframe; t++)
  {
    real inputMax = -THInf;
    for(d = 0; d < dim; d++) {
      if (input_data[d] >= inputMax) inputMax = input_data[d];
    }

    accreal sum = 0;
    for(d = 0; d < dim; d++) {
      real z = THExpMinusApprox(inputMax - input_data[d]);
      output_data[d] = z;
      sum += z;
    }

    for(d = 0; d < dim; d++) {
      output_data[d] *= 1/sum;
    }

    input_data += dim;
    output_data += dim;
  }

  THTensor_(free)(input);

  return 1;
}

static int nn_(SoftMax_updateGradInput)(lua_State *L)
{
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  real *gradInput_data, *gradOutput_data, *output_data;
  long nframe = 0, dim = 0;
  long t, d;

  if(output->nDimension == 1)
  {
    nframe = 1;
    dim = output->size[0];
  }
  else if(output->nDimension == 2)
  {
    nframe = output->size[0];
    dim = output->size[1];
  }
  else
    THError("vector or matrix expected");

  THTensor_(resizeAs)(gradInput, output);
  gradInput_data = THTensor_(data)(gradInput);
  output_data = THTensor_(data)(output);
  gradOutput_data = THTensor_(data)(gradOutput);
  for(t = 0; t < nframe; t++)
  {
    accreal sum = 0;
    for(d = 0; d < dim; d++)
      sum += (accreal)gradOutput_data[d] * output_data[d];

    for(d = 0; d < dim; d++)
      gradInput_data[d] = output_data[d] * (gradOutput_data[d] - sum);

    gradInput_data += dim;
    output_data += dim;
    gradOutput_data += dim;
  }

  return 1;
}

static const struct luaL_Reg nn_(SoftMax__) [] = {
  {"SoftMax_updateOutput", nn_(SoftMax_updateOutput)},
  {"SoftMax_updateGradInput", nn_(SoftMax_updateGradInput)},
  {NULL, NULL}
};

static void nn_(SoftMax_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SoftMax__), "nn");
  lua_pop(L,1);
}

#endif
