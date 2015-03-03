#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricMaxPooling.c"
#else

static void nn_(VolumetricMaxPooling_updateOutput_frame)(real *input_p, real *output_p,
							 real *indx_p, real *indy_p, real *indz_p,
							 long nslices,
							 long itime, long iwidth, long iheight,
							 long otime, long owidth, long oheight,
							 int kT, int kW, int kH, int dT, int dW, int dH)
{
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    /* loop over output */
    long i, j, ti;
    for(ti = 0; ti < otime; ti++)
    {
      for(i = 0; i < oheight; i++)
      {
	for(j = 0; j < owidth; j++)
	{
	  /* local pointers */
	  real *ip = input_p   + k*itime*iwidth*iheight + ti*iwidth*iheight*dT +  i*iwidth*dH + j*dW;
	  real *op = output_p  + k*otime*owidth*oheight + ti*owidth*oheight + i*owidth + j;
	  real *indzp = indz_p + k*otime*owidth*oheight + ti*owidth*oheight + i*owidth + j;
	  real *indyp = indy_p + k*otime*owidth*oheight + ti*owidth*oheight + i*owidth + j;
	  real *indxp = indx_p + k*otime*owidth*oheight + ti*owidth*oheight + i*owidth + j;
	  
	  /* compute local max: */
	  real maxval = -THInf;
	  int x,y,z;

	  *indzp = -1;
	  *indyp = -1;
	  *indxp = -1;
	  for(z=0; z < kT; z++)
	  {
	    for(y = 0; y < kH; y++)
	    {
	      for(x = 0; x < kW; x++)
	      {
		real val = *(ip + z*iwidth*iheight + y*iwidth + x);
		if (val > maxval)
		{
		  maxval = val;
		  *indzp = z+1;
		  *indyp = y+1;
		  *indxp = x+1;
		}
	      }
	    }
	  }

	  /* set output to local max */
	  *op = maxval;
	  
	  /* store location of max (x,y) */
	  /**indyp = (int)(maxindex / kW)+1;*/
	  /**indxp = (maxindex % kW) +1;*/
	}
      }
    }
  }
}

static int nn_(VolumetricMaxPooling_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int kT = luaT_getfieldcheckint(L, 1, "kT");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dT = luaT_getfieldcheckint(L, 1, "dT");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  long nslices;
  long itime;
  long iheight;
  long iwidth;
  long otime;
  long oheight;
  long owidth;
  real *input_data;
  real *output_data;
  real *indices_data;

  luaL_argcheck(L, input->nDimension == 4 || input->nDimension == 5, 2, "4D or 5D (batch-mode) tensor expected");

  int dimN = 0;
  int dimt = 1;
  int dimh = 2;
  int dimw = 3;

  if (input->nDimension == 5) {
    dimN++;
    dimt++;
    dimh++;
    dimw++;
  }

  luaL_argcheck(L, input->size[dimw] >= kW && input->size[dimh] >= kH && input->size[dimt] >= kT, 2, "input image smaller than kernel size");

  /* sizes */
  nslices = input->size[dimN];
  itime = input->size[dimt];
  iheight = input->size[dimh];
  iwidth = input->size[dimw];
  otime = (itime - kT) / dT + 1;
  oheight = (iheight - kH) / dH + 1;
  owidth = (iwidth - kW) / dW + 1;

  /* get contiguous input */
  input = THTensor_(newContiguous)(input);

  if (input->nDimension == 4) { /* non-batch mode */
    /* resize output */
    THTensor_(resize4d)(output, nslices, otime, oheight, owidth);
    /* indices will contain ti,i,j locations for each output point */
    THTensor_(resize5d)(indices, 3, nslices, otime, oheight, owidth);
    
    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THTensor_(data)(indices);
    
    nn_(VolumetricMaxPooling_updateOutput_frame)(input_data, output_data,
  					       indices_data+nslices*otime*owidth*oheight*2, 
  					       indices_data+nslices*otime*owidth*oheight, 
  					       indices_data,
  					       nslices,
  					       itime, iwidth, iheight,
  					       otime, owidth, oheight,
  					       kT, kW, kH, dT, dW, dH);
  }
  else { /* batch mode */
    long p;
    long nBatch = input->size[0];

    long istride = nslices*itime*iwidth*iheight;
    long ostride = nslices*otime*owidth*oheight;

    /* resize output */
    THTensor_(resize5d)(output, nBatch, nslices, otime, oheight, owidth);
    /* indices will contain ti,i,j locations for each output point */

    THLongStorage* size = THLongStorage_newWithSize(6);
    size->data[0] = 3; size->data[1] = nBatch;
    size->data[2] = nslices; size->data[3] = otime; 
    size->data[4] = oheight; size->data[5] = owidth;
    THTensor_(resize)(indices, size, NULL); /* resize6d not available */
    //TODO: Replace with resize6d when available
    //THTensor_(resize6d)(indices, 3, nBatch, nslices, otime, oheight, owidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THTensor_(data)(indices);

#pragma omp parallel for private(p)
    for (p=0; p < nBatch; p++) 
    {
      nn_(VolumetricMaxPooling_updateOutput_frame)(
                    input_data+p*istride, 
                    output_data+p*ostride,
                    indices_data+(p+nBatch+nBatch)*ostride, 
                    indices_data+(p+nBatch)*ostride, 
                    indices_data+p*ostride,
                    nslices,
                    itime, iwidth, iheight,
                    otime, owidth, oheight,
                    kT, kW, kH, dT, dW, dH);
    }

    THLongStorage_free(size);
  }

  /* cleanup */
  THTensor_(free)(input);
  return 1;
}

static void nn_(VolumetricMaxPooling_updateGradInput_frame)(real *gradInput_p, real *gradOutput_p,
							    real *indx_p, real *indy_p, real *indz_p,
							    long nslices,
							    long itime, long iwidth, long iheight,
							    long otime, long owidth, long oheight,
							    int dT, int dW, int dH)
{
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    real *gradInput_p_k = gradInput_p + k*itime*iwidth*iheight;
    real *gradOutput_p_k = gradOutput_p + k*otime*owidth*oheight;
    real *indx_p_k = indx_p + k*otime*owidth*oheight;
    real *indy_p_k = indy_p + k*otime*owidth*oheight;
    real *indz_p_k = indz_p + k*otime*owidth*oheight;

    /* calculate max points */
    long ti, i, j;
    for(ti = 0; ti < otime; ti++)
    {
      for(i = 0; i < oheight; i++)
      {
	for(j = 0; j < owidth; j++)
	{
	  /* retrieve position of max */
	  long maxti = indz_p_k[ti*oheight*owidth + i*owidth + j] - 1 + ti*dT;
	  long maxi  = indy_p_k[ti*oheight*owidth + i*owidth + j] - 1 + i*dH;
	  long maxj  = indx_p_k[ti*oheight*owidth + i*owidth + j] - 1 + j*dW;
	  
	  /* update gradient */
	  gradInput_p_k[maxti*iheight*iwidth + maxi*iwidth + maxj] += gradOutput_p_k[ti*oheight*owidth + i*owidth + j];
	}
      }
    }
  }
}

static int nn_(VolumetricMaxPooling_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  int dT = luaT_getfieldcheckint(L, 1, "dT");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  int nslices;
  int itime;
  int iheight;
  int iwidth;
  int otime;
  int oheight;
  int owidth;
  real *gradInput_data;
  real *gradOutput_data;
  real *indices_data;

  int dimN = 0;
  int dimt = 1;
  int dimh = 2;
  int dimw = 3;


  /* get contiguous gradOutput */
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* resize */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  if (input->nDimension == 5) {
    dimN++;
    dimt++;
    dimh++;
    dimw++;
  }

  /* sizes */
  nslices = input->size[dimN];
  itime = input->size[dimt];
  iheight = input->size[dimh];
  iwidth = input->size[dimw];
  otime = gradOutput->size[dimt];
  oheight = gradOutput->size[dimh];
  owidth = gradOutput->size[dimw];

  /* get raw pointers */
  gradInput_data = THTensor_(data)(gradInput);
  gradOutput_data = THTensor_(data)(gradOutput);
  indices_data = THTensor_(data)(indices);

  /* backprop */
  if (input->nDimension == 4) { /* non-batch mode*/

    nn_(VolumetricMaxPooling_updateGradInput_frame)(gradInput_data, gradOutput_data,
  						  indices_data+nslices*otime*owidth*oheight*2, 
  						  indices_data+nslices*otime*owidth*oheight, 
  						  indices_data,
  						  nslices,
  						  itime, iwidth, iheight,
  						  otime, owidth, oheight,
  						  dT, dW, dH);
  }
  else { /* batch mode */
    long p;
    long nBatch = input->size[0];

    long istride = nslices*itime*iwidth*iheight;
    long ostride = nslices*otime*owidth*oheight;
    
#pragma omp parallel for private(p)
    for (p = 0; p < nBatch; p++)
    {
      nn_(VolumetricMaxPooling_updateGradInput_frame)(
                gradInput_data+p*istride, 
                gradOutput_data+p*ostride,
                indices_data+(p+nBatch+nBatch)*ostride, 
                indices_data+(p+nBatch)*ostride,
                indices_data+p*ostride,
                nslices,
                itime, iwidth, iheight,
                otime, owidth, oheight,
                dT, dW, dH);
    }
  }

  /* cleanup */
  THTensor_(free)(gradOutput);
  return 1;
}

static const struct luaL_Reg nn_(VolumetricMaxPooling__) [] = {
  {"VolumetricMaxPooling_updateOutput", nn_(VolumetricMaxPooling_updateOutput)},
  {"VolumetricMaxPooling_updateGradInput", nn_(VolumetricMaxPooling_updateGradInput)},
  {NULL, NULL}
};

static void nn_(VolumetricMaxPooling_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(VolumetricMaxPooling__), "nn");
  lua_pop(L,1);
}

#endif
