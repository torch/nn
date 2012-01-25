#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialSubSampling.c"
#else

static int nn_(SpatialSubSampling_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  real *weight_data = THTensor_(data)(weight);
  real *bias_data = THTensor_(data)(bias);
  real *output_data;
  real *input_data;


  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D(batch mode) tensor expected");

  int dimw = 2;
  int dimh = 1;
  if (input->nDimension == 4) {
    dimw++;
    dimh++;
  }

  long inputWidth = input->size[dimw];
  long inputHeight = input->size[dimh];
  long outputWidth = (inputWidth - kW) / dW + 1;
  long outputHeight = (inputHeight - kH) / dH + 1;


  luaL_argcheck(L, input->size[dimh-1] == nInputPlane, 2, "invalid number of input planes");
  luaL_argcheck(L, inputWidth >= kW && inputHeight >= kH, 2, "input image smaller than kernel size");

  input = THTensor_(newContiguous)(input);
  input_data = THTensor_(data)(input);

  long nbatch = 1;
  if (input->nDimension == 3) 
  {
    THTensor_(resize3d)(output, nInputPlane, outputHeight, outputWidth);
  }
  else
  {
    nbatch = input->size[0];
    THTensor_(resize4d)(output, nbatch, nInputPlane, outputHeight, outputWidth);
  }

  output_data = THTensor_(data)(output);

  long i, k, p;

  for(p = 0; p < nbatch; p++)
  {
    //input_data += p*nInputPlane*inputWidth*inputHeight;
    //output_data += p*nInputPlane*outputHeight*outputWidth;
    for(k = 0; k < nInputPlane; k++)
    {
      real *ptr_output;
      long xx, yy;

      /* Get the good mask for (k,i) (k out, i in) */
      real the_weight = weight_data[k];

      /* Initialize to the bias */
      real z = bias_data[k];
      for(i = 0; i < outputWidth*outputHeight; i++)
	output_data[i] = z;
      
      /* For all output pixels... */
      ptr_output = output_data;
      for(yy = 0; yy < outputHeight; yy++)
      {
	for(xx = 0; xx < outputWidth; xx++)
	{
	  // Compute the mean of the input image...
	  real *ptr_input = input_data+yy*dH*inputWidth+xx*dW;
	  real sum = 0;
	  long kx, ky;

	  for(ky = 0; ky < kH; ky++)
	  {
	    for(kx = 0; kx < kW; kx++)
	      sum += ptr_input[kx];
	    ptr_input += inputWidth; // next input line
	  }
	  
	  // Update output
	  *ptr_output++ += the_weight*sum;
	}
      }

      // Next input/output plane
      output_data += outputWidth*outputHeight;
      input_data += inputWidth*inputHeight;
    }
  }

  THTensor_(free)(input);

  return 1;
}

static int nn_(SpatialSubSampling_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));
  
  int dimw = 2;
  int dimh = 1;
  long nbatch = 1;
  if (input->nDimension == 4) {
    dimw++;
    dimh++;
    nbatch = input->size[0];
  }

  long inputWidth = input->size[dimw];
  long inputHeight = input->size[dimh];
  long outputWidth = (inputWidth - kW) / dW + 1;
  long outputHeight = (inputHeight - kH) / dH + 1;

  real *weight_data = THTensor_(data)(weight);
  real *gradOutput_data = THTensor_(data)(gradOutput);
  real *gradInput_data;

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);  
  gradInput_data = THTensor_(data)(gradInput);
  gradOutput_data = THTensor_(data)(gradOutput);

  long i, k, p;

  for(p = 0; p < nbatch; p++)
  {
    //gradInput_data += p*nInputPlane*inputWidth*inputHeight;
    //gradOutput_data += p*nInputPlane*outputWidth*outputHeight;
    for(k = 0; k < nInputPlane; k++)
    {
      real the_weight = weight_data[k];
      real *ptr_gradOutput = gradOutput_data;
      long xx, yy;
      
      for(yy = 0; yy < outputHeight; yy++)
      {
	for(xx = 0; xx < outputWidth; xx++)
	{
	  real *ptr_gradInput = gradInput_data+yy*dH*inputWidth+xx*dW;
	  real z = *ptr_gradOutput++ * the_weight;
	  long kx, ky;
	  
	  for(ky = 0; ky < kH; ky++)
	  {
	    for(kx = 0; kx < kW; kx++)
	      ptr_gradInput[kx] += z;
	    ptr_gradInput += inputWidth;
	  }    
	}
      }
      gradOutput_data += outputWidth*outputHeight;
      gradInput_data += inputWidth*inputHeight;
    }
  }

  return 1;
}

static int nn_(SpatialSubSampling_accGradParameters)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));  
  real scale = luaL_optnumber(L, 4, 1);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");

  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_(Tensor_id));
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_(Tensor_id));
  
  int dimw = 2;
  int dimh = 1;
  long nbatch = 1;
  if (input->nDimension == 4) {
    dimw++;
    dimh++;
    nbatch = input->size[0];
  }

  long inputWidth = input->size[dimw];
  long inputHeight = input->size[dimh];
  long outputWidth = (inputWidth - kW) / dW + 1;
  long outputHeight = (inputHeight - kH) / dH + 1;

  real *gradWeight_data = THTensor_(data)(gradWeight);
  real *gradBias_data = THTensor_(data)(gradBias);
  real *gradOutput_data = THTensor_(data)(gradOutput);
  real *input_data;

  input = THTensor_(newContiguous)(input);
  input_data = THTensor_(data)(input);

  long i, k, p;
  for(p = 0; p < nbatch; p++)
  {
    //input_data += p*nInputPlane*inputWidth*inputHeight;
    //gradOutput_data += p*nInputPlane*inputWidth*inputHeight;
    for(k = 0; k < nInputPlane; k++)
    {
      real *ptr_gradOutput = gradOutput_data;
      real sum;
      long xx, yy;

      sum = 0;
      for(i = 0; i < outputWidth*outputHeight; i++)
	sum += gradOutput_data[i];
      gradBias_data[k] += scale*sum;

      sum = 0;
      for(yy = 0; yy < outputHeight; yy++)
      {
	for(xx = 0; xx < outputWidth; xx++)
	{
	  real *ptr_input = input_data+yy*dH*inputWidth+xx*dW;
	  real z = *ptr_gradOutput++;
	  long kx, ky;

	  for(ky = 0; ky < kH; ky++)
	  {
	    for(kx = 0; kx < kW; kx++)
	      sum += z * ptr_input[kx];
	    ptr_input += inputWidth;
	  }    
	}
      }
      gradWeight_data[k] += scale*sum;
      gradOutput_data += outputWidth*outputHeight;
      input_data += inputWidth*inputHeight;
    }
  }


  THTensor_(free)(input);

  return 0;
}

static const struct luaL_Reg nn_(SpatialSubSampling__) [] = {
  {"SpatialSubSampling_updateOutput", nn_(SpatialSubSampling_updateOutput)},
  {"SpatialSubSampling_updateGradInput", nn_(SpatialSubSampling_updateGradInput)},
  {"SpatialSubSampling_accGradParameters", nn_(SpatialSubSampling_accGradParameters)},
  {NULL, NULL}
};

static void nn_(SpatialSubSampling_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(SpatialSubSampling__), "nn");
  lua_pop(L,1);
}

#endif
