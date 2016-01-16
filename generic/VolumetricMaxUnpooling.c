#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricMaxUnpooling.c"
#else

static void nn_(VolumetricMaxUnpooling_updateOutput_frame)(real *input_p, real *output_p,
                                                      real *ind_p,
                                                      long nslices,
                                                      long itime, long iwidth, long iheight,
                                                      long otime, long owidth, long oheight,
                                                      int dT, int dW, int dH)
{
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {    
    long ti, i, j, maxz, maxy, maxx;
    for(ti = 0; ti < itime; ti++)
    {
      for(i = 0; i < iheight; i++)
      {
        for(j = 0; j < iwidth; j++)
        {
          real *output_p_k = output_p + k*otime*owidth*oheight + ti*owidth*oheight*dT + i*owidth*dH + j*dW;
          real *input_p_k = input_p + k*itime*iwidth*iheight + ti*iwidth*iheight + i*iwidth + j;
          real *ind_p_k = ind_p + k*itime*iwidth*iheight + ti*iwidth*iheight + i*iwidth + j;
          
          maxz = ((unsigned char*)(ind_p_k))[0]; /* retrieve position of max */
          maxy = ((unsigned char*)(ind_p_k))[1];
          maxx = ((unsigned char*)(ind_p_k))[2];

          if(maxz<0 || maxy<0 || maxx<0 || maxz>=otime || maxy>=oheight || maxx>=owidth)
          {
              THError("invalid max index maxz= %d, maxy= %d, maxx= %d, otime= %d, owidth= %d, oheight= %d", maxz, maxy, maxx, otime, owidth, oheight);
          }
          output_p_k[oheight*owidth*maxz + owidth*maxy + maxx] = *input_p_k; /* update output */
        }
      }
    }
  }
}

static int nn_(VolumetricMaxUnpooling_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  int otime = luaT_getfieldcheckint(L, 1, "otime");
  int owidth = luaT_getfieldcheckint(L, 1, "owidth");
  int oheight = luaT_getfieldcheckint(L, 1, "oheight");
  int dT = luaT_getfieldcheckint(L, 1, "dT");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dimw = 3;
  int dimh = 2;
  int dimt = 1;
  int nbatch = 1;
  int nslices;
  int itime;
  int iheight;
  int iwidth;
  real *input_data;
  real *output_data;
  real *indices_data;

  luaL_argcheck(L, input->nDimension == 4 || input->nDimension == 5 , 2, "4D or 5D (batch mode) tensor expected");
  if (!THTensor_(isSameSizeAs)(input, indices)){
    THError("Invalid input size w.r.t current indices size");
  }  

  if (input->nDimension == 5) 
  {
    nbatch = input->size[0];
    dimt++;
    dimw++;
    dimh++;
  }

  /* sizes */
  nslices = input->size[dimt-1];
  itime = input->size[dimt];
  iheight = input->size[dimh];
  iwidth = input->size[dimw];

  /* get contiguous input */
  input = THTensor_(newContiguous)(input);
  indices = THTensor_(newContiguous)(indices);

  /* resize output */
  if (input->nDimension == 4)
  {
    THTensor_(resize4d)(output, nslices, otime, oheight, owidth);
    THTensor_(zero)(output);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THTensor_(data)(indices);

    nn_(VolumetricMaxUnpooling_updateOutput_frame)(input_data, output_data,
                                              indices_data,
                                              nslices,
                                              itime, iwidth, iheight,
                                              otime, owidth, oheight,
                                              dT, dW, dH);
  }
  else
  {
    long p;

    THTensor_(resize5d)(output, nbatch, nslices, otime, oheight, owidth);
    THTensor_(zero)(output);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THTensor_(data)(indices);

#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      nn_(VolumetricMaxUnpooling_updateOutput_frame)(input_data+p*nslices*itime*iwidth*iheight, output_data+p*nslices*otime*owidth*oheight,
                                                indices_data+p*nslices*itime*iwidth*iheight,
                                                nslices,
                                                itime, iwidth, iheight,
                                                otime, owidth, oheight,
                                                dT, dW, dH);
    }
  }

  /* cleanup */
  THTensor_(free)(input);
  THTensor_(free)(indices);
  return 1;
}

static void nn_(VolumetricMaxUnpooling_updateGradInput_frame)(real *gradInput_p, real *gradOutput_p,
                                                         real *ind_p,
                                                         long nslices,
                                                         long itime, long iwidth, long iheight,
                                                         long otime, long owidth, long oheight,
                                                         int dT, int dW, int dH)
{
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    long ti, i, j, maxz, maxy, maxx;
    for(ti = 0; ti < itime; ti++)
    {
      for(i = 0; i < iheight; i++)
      {
        for(j = 0; j < iwidth; j++)
        {        
          real *gradInput_p_k = gradInput_p + k*itime*iwidth*iheight + ti*iwidth*iheight + i*iwidth + j;
          real *gradOutput_p_k = gradOutput_p + k*otime*owidth*oheight + ti*owidth*oheight*dT + i*owidth*dH + j*dW;
          real *ind_p_k = ind_p + k*itime*iwidth*iheight + ti*iwidth*iheight + i*iwidth + j;
          
          maxz = ((unsigned char*)(ind_p_k))[0]; /* retrieve position of max */
          maxy = ((unsigned char*)(ind_p_k))[1];
          maxx = ((unsigned char*)(ind_p_k))[2];

          if(maxz<0 || maxy<0 || maxx<0 || maxz>=otime || maxy>=oheight || maxx>=owidth)
          {
              THError("invalid max index maxz= %d, maxy= %d, maxx= %d, otime= %d, owidth= %d, oheight= %d", maxz, maxy, maxx, otime, owidth, oheight);
          }  
          *gradInput_p_k = gradOutput_p_k[oheight*owidth*maxz + owidth*maxy + maxx]; /* update gradient */
        }
      }
    }
  }
}

static int nn_(VolumetricMaxUnpooling_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  int otime = luaT_getfieldcheckint(L, 1, "otime");
  int owidth = luaT_getfieldcheckint(L, 1, "owidth");
  int oheight = luaT_getfieldcheckint(L, 1, "oheight");
  int dT = luaT_getfieldcheckint(L, 1, "dT");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dimw = 3;
  int dimh = 2;
  int dimt = 1;
  int nbatch = 1;
  int nslices;
  int itime;
  int iheight;
  int iwidth;
  real *gradInput_data;
  real *gradOutput_data;
  real *indices_data;

  if (!THTensor_(isSameSizeAs)(input, indices)){
    THError("Invalid input size w.r.t current indices size");
  } 

  /* get contiguous gradOutput */
  gradOutput = THTensor_(newContiguous)(gradOutput);
  indices = THTensor_(newContiguous)(indices);

  /* resize */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  if (input->nDimension == 5) {
    nbatch = input->size[0];
    dimt++;
    dimw++;
    dimh++;
  }

  /* sizes */
  nslices = input->size[dimt-1];
  itime = input->size[dimt];
  iheight = input->size[dimh];
  iwidth = input->size[dimw];

  if(otime!=gradOutput->size[dimt] || owidth!=gradOutput->size[dimw] || oheight!=gradOutput->size[dimh]){
    THError("Inconsistent gradOutput size. otime= %d, oheight= %d, owidth= %d, gradOutput: %dx%d", otime, oheight, owidth,gradOutput->size[dimh],gradOutput->size[dimw]);
  }

  /* get raw pointers */
  gradInput_data = THTensor_(data)(gradInput);
  gradOutput_data = THTensor_(data)(gradOutput);
  indices_data = THTensor_(data)(indices);

  /* backprop */
  if (input->nDimension == 4)
  {
    nn_(VolumetricMaxUnpooling_updateGradInput_frame)(gradInput_data, gradOutput_data,
                                                 indices_data,
                                                 nslices,
                                                 itime, iwidth, iheight,
                                                 otime, owidth, oheight,
                                                 dT, dW, dH);
  }
  else
  {
    long p;
#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      nn_(VolumetricMaxUnpooling_updateGradInput_frame)(gradInput_data+p*nslices*itime*iwidth*iheight, gradOutput_data+p*nslices*otime*owidth*oheight,
                                                   indices_data+p*nslices*itime*iwidth*iheight,
                                                   nslices,
                                                   itime, iwidth, iheight,
                                                   otime, owidth, oheight,
                                                   dT, dW, dH);
    }
  }

  /* cleanup */
  THTensor_(free)(gradOutput);
  THTensor_(free)(indices);

  return 1;
}

static const struct luaL_Reg nn_(VolumetricMaxUnpooling__) [] = {
  {"VolumetricMaxUnpooling_updateOutput", nn_(VolumetricMaxUnpooling_updateOutput)},
  {"VolumetricMaxUnpooling_updateGradInput", nn_(VolumetricMaxUnpooling_updateGradInput)},
  {NULL, NULL}
};

static void nn_(VolumetricMaxUnpooling_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(VolumetricMaxUnpooling__), "nn");
  lua_pop(L,1);
}

#endif
