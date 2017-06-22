#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricConvolutionMM.c"
#else

static void inline THNN_(VolumetricConvolutionMM_shapeCheck)(
                         THNNState *state,
                         THTensor *input,
                         THTensor *gradOutput,
                         THTensor *weight,
                         THTensor *bias,
                         int kT,
                         int kW,
                         int kH,
                         int dT,
                         int dW,
                         int dH,
                         int pT,
                         int pW,
                         int pH) {
  THNN_ARGCHECK(input->nDimension == 4 || input->nDimension == 5, 2, input,
                "4D or 5D (batch mode) tensor expected for input, but got: %s");
  THArgCheck(kT > 0 && kW > 0 && kH > 0, 8,
             "kernel size should be greater than zero, but got kT: %d kH: %d kW: %d", kT, kH, kW);
  THArgCheck(dT > 0 && dW > 0 && dH > 0, 11,
             "stride should be greater than zero, but got dT: %d dH: %d dW: %d", dT, dH, dW);

  int ndim = input->nDimension;
  int dimf = 0;
  int dimt = 1;
  int dimh = 2;
  int dimw = 3;

  if (ndim == 5)
  {
    dimf++;
    dimt++;
    dimh++;
    dimw++;
  }

  long nInputPlane;
  long inputDepth;
  long inputHeight;
  long inputWidth;
  long nOutputPlane;
  long outputDepth;
  long outputHeight;
  long outputWidth;

  nInputPlane = input->size[dimf];
  inputDepth = input->size[dimt];
  inputHeight  = input->size[dimh];
  inputWidth   = input->size[dimw];
  nOutputPlane = weight->size[0];
  outputDepth  = (inputDepth + 2*pT - kT) / dT + 1;
  outputHeight = (inputHeight + 2*pH - kH) / dH + 1;
  outputWidth  = (inputWidth + 2*pW - kW) / dW + 1;

  if (outputWidth < 1 || outputHeight < 1 || outputDepth < 1)
  {
    THError(
      "Given input size: (%dx%dx%dx%d). Calculated output size: (%dx%dx%dx%d). Output size is too small",
      nInputPlane, inputDepth, inputHeight, inputWidth,
      nOutputPlane, outputDepth, outputHeight, outputWidth
    );
  }

  THArgCheck(weight->nDimension == 2 || weight->nDimension == 5, 4,
             "weight tensor should be 2D or 5D - got %d", weight->nDimension);

  if (bias != NULL) {
    THNN_CHECK_DIM_SIZE(bias, 1, 0, weight->size[0]);
  }

  THNN_CHECK_DIM_SIZE(input, ndim, dimf, nInputPlane);

  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimf, nOutputPlane);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimt, outputDepth);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimh, outputHeight);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimw, outputWidth);
  }
}

static int THNN_(view_weight)(THTensor **_weight)
{
  THTensor *weight = *_weight;
  if (weight->nDimension == 5) {
    long s1 = weight->size[0];
    long s2 = weight->size[1] * weight->size[2] * weight->size[3] * weight->size[4];
    *_weight = THTensor_(newWithStorage2d)(weight->storage, weight->storageOffset, s1, -1, s2, -1);
    return 1;
  }
  return 0;
}

/* note: due to write issues, this one cannot be parallelized as well as unfolded_copy */
static void THNN_(unfolded_acc_vol)(
          THTensor *finput,
          THTensor *input,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH,
          int nInputPlane,
          int inputDepth,
          int inputWidth,
          int inputHeight,
          int outputDepth,
          int outputWidth,
          int outputHeight)
{
  int nip;
  real *input_data = THTensor_(data)(input);
  real *finput_data = THTensor_(data)(finput);

//#pragma omp parallel for private(nip)
  for (nip = 0; nip < nInputPlane; nip++)
  {
    int kt, kw, kh, t, y, x, it, ix, iy;
    for (kt = 0; kt < kT; kt++)
    {
      for (kh = 0; kh < kH; kh++)
      {
        for (kw = 0; kw < kW; kw++)
        {
          real *src = finput_data
            + nip * (kT*kH*kW*outputDepth*outputHeight*outputWidth)
            + kt  * (kH*kW*outputDepth*outputHeight*outputWidth)
            + kh  * (kW*outputDepth*outputHeight*outputWidth)
            + kw  * (outputDepth*outputHeight*outputWidth);

          real *dst = input_data + nip*(inputDepth*inputHeight*inputWidth);
          if (pT > 0 || pH > 0 || pW > 0)
          {
            for (t = 0; t < outputDepth; t++)
            {
              it = t*dT - pT + kt;
              for (y = 0; y < outputHeight; y++)
              {
                iy = y*dH - pH + kh;
                for (x = 0; x < outputWidth; x++)
                {
                  ix = x*dW - pW + kw;
                  if (it < 0 || it >= inputDepth || iy < 0 || iy >= inputHeight || ix < 0 || ix >= inputWidth)
                  {
                  }
                  else
                  {
                    real *dst_slice = dst+it*inputHeight*inputWidth+iy*inputWidth+ix;
                    THVector_(cadd)(dst_slice, dst_slice, src+t*outputHeight*outputWidth+y*outputWidth+x, 1, 1);
                  }
                }
              }
            }
          }
          else
          {
            for (t = 0; t < outputDepth; t++)
            {
              it = t*dT + kt;
              for (y = 0; y < outputHeight; y++)
              {
                iy = y*dH + kh;
                for(x = 0; x < outputWidth; x++)
                {
                  ix = x*dW + kw;
                  real *dst_slice = dst+it*inputHeight*inputWidth+iy*inputWidth+ix;
                  THVector_(cadd)(dst_slice, dst_slice, src+t*outputHeight*outputWidth+y*outputWidth+x, 1, 1);
                }
              }
            }
          }
        }
      }
    }
  }
}

static void THNN_(unfolded_copy_vol)(
          THTensor *finput,
          THTensor *input,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH,
          int nInputPlane,
          int inputDepth,
          int inputWidth,
          int inputHeight,
          int outputDepth,
          int outputWidth,
          int outputHeight)
{
  long k;
  real *input_data = THTensor_(data)(input);
  real *finput_data = THTensor_(data)(finput);
// #pragma omp parallel for private(k)
  for (k = 0; k < nInputPlane*kT*kH*kW; k++)
  {
    int nip = k / (kT*kH*kW);
    int rest = k % (kT*kH*kW);
    int kt = rest / (kH*kW);
    rest = rest % (kH*kW);
    int kh = rest / kW;
    int kw = rest % kW;
    int t,x,y,it,ix,iy;
    real *dst = finput_data
      + nip * (kT*kH*kW*outputDepth*outputHeight*outputWidth)
      + kt  * (kH*kW*outputDepth*outputHeight*outputWidth)
      + kh  * (kW*outputDepth*outputHeight*outputWidth)
      + kw  * (outputDepth*outputHeight*outputWidth);
    real *src = input_data + nip*(inputDepth*inputHeight*inputWidth);

    if (pT > 0 || pH > 0 || pW > 0)
    {
      for (t = 0; t < outputDepth; t++)
      {
        it = t*dT - pT + kt;
        for (y = 0; y < outputHeight; y++)
        {
          iy = y*dH - pH + kh;
          for (x = 0; x < outputWidth; x++)
          {
            ix = x*dW - pW + kw;
            if (it < 0 || it >= inputDepth || iy < 0 || iy >= inputHeight || ix < 0 || ix >= inputWidth)
              memset(dst+t*outputHeight*outputWidth+y*outputWidth+x, 0, sizeof(real)*(1));
            else
              memcpy(dst+t*outputHeight*outputWidth+y*outputWidth+x, src+it*inputHeight*inputWidth+iy*inputWidth+ix, sizeof(real)*(1));
          }
        }
      }
    }
    else
    {
      for (t = 0; t < outputDepth; t++)
      {
        it = t*dT + kt;
        for (y = 0; y < outputHeight; y++)
        {
          iy = y*dH + kh;
          for(x = 0; x < outputWidth; x++)
          {
            ix = x*dW + kw;
            memcpy(dst+t*outputHeight*outputWidth+y*outputWidth+x, src+it*inputHeight*inputWidth+iy*inputWidth+ix, sizeof(real)*(1));
          }
        }
      }
    }
  }
}

static void THNN_(VolumetricConvolutionMM_updateOutput_frame)(
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH,
          long nInputPlane,
          long inputDepth,
          long inputWidth,
          long inputHeight,
          long nOutputPlane,
          long outputDepth,
          long outputWidth,
          long outputHeight)
{
  long i;
  THTensor *output2d;

  THNN_(unfolded_copy_vol)(
    finput, input,
    kT, kW, kH,
    dT, dW, dH,
    pT, pW, pH,
    nInputPlane,
    inputDepth, inputWidth, inputHeight,
    outputDepth, outputWidth, outputHeight
  );

  output2d = THTensor_(newWithStorage2d)(
    output->storage, output->storageOffset, nOutputPlane, -1,
    outputDepth*outputHeight*outputWidth, -1
  );

  if (bias) {
      for (i = 0; i < nOutputPlane; i++)
      {
        THVector_(fill)(
          output->storage->data+output->storageOffset+output->stride[0]*i,
          THTensor_(get1d)(bias, i),
          outputDepth*outputHeight*outputWidth
        );
      }
  } else {
    THTensor_(zero)(output);
  }

  THTensor_(addmm)(output2d, 1, output2d, 1, weight, finput);

  THTensor_(free)(output2d);
}

void THNN_(VolumetricConvolutionMM_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH)
{
  int dimf = 0;
  int dimt = 1;
  int dimh = 2;
  int dimw = 3;
  int freeWeight = 0;

  long nInputPlane;
  long inputDepth;
  long inputHeight;
  long inputWidth;
  long nOutputPlane;
  long outputDepth;
  long outputHeight;
  long outputWidth;

  THNN_(VolumetricConvolutionMM_shapeCheck)(
        state, input, NULL, weight, bias,
        kT, kW, kH, dT, dW, dH, pT, pW, pH);
  input = THTensor_(newContiguous)(input);

  if (input->nDimension == 5)
  {
    dimf++;
    dimt++;
    dimh++;
    dimw++;
  }

  nInputPlane = input->size[dimf];
  inputDepth = input->size[dimt];
  inputHeight  = input->size[dimh];
  inputWidth   = input->size[dimw];
  nOutputPlane = weight->size[0];
  outputDepth  = (inputDepth + 2*pT - kT) / dT + 1;
  outputHeight = (inputHeight + 2*pH - kH) / dH + 1;
  outputWidth  = (inputWidth + 2*pW - kW) / dW + 1;

  freeWeight = THNN_(view_weight)(&weight);

  if (input->nDimension == 4)
  {
    THTensor_(resize2d)(finput, kT*kW*kH*nInputPlane, outputDepth*outputHeight*outputWidth);
    THTensor_(resize4d)(output, nOutputPlane, outputDepth, outputHeight, outputWidth);

    THNN_(VolumetricConvolutionMM_updateOutput_frame)(
      input, output, weight, bias, finput,
      kT, kW, kH,
      dT, dW, dH,
      pT, pW, pH,
      nInputPlane, inputDepth, inputWidth, inputHeight,
      nOutputPlane, outputDepth, outputWidth, outputHeight
    );
  }
  else
  {
    long T = input->size[0];
    long t;

    THTensor_(resize3d)(finput, T, kT*kW*kH*nInputPlane, outputDepth*outputHeight*outputWidth);
    THTensor_(resize5d)(output, T, nOutputPlane, outputDepth, outputHeight, outputWidth);

// #pragma omp parallel for private(t)
    for (t = 0; t < T; t++)
    {
      THTensor *input_t = THTensor_(newSelect)(input, 0, t);
      THTensor *output_t = THTensor_(newSelect)(output, 0, t);
      THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);

      THNN_(VolumetricConvolutionMM_updateOutput_frame)(
        input_t, output_t, weight, bias, finput_t,
        kT, kW, kH,
        dT, dW, dH,
        pT, pW, pH,
        nInputPlane, inputDepth, inputWidth, inputHeight,
        nOutputPlane, outputDepth, outputWidth, outputHeight
      );

      THTensor_(free)(input_t);
      THTensor_(free)(output_t);
      THTensor_(free)(finput_t);
    }
  }

  THTensor_(free)(input);
  if (freeWeight)
    THTensor_(free)(weight);
}

static void THNN_(VolumetricConvolutionMM_updateGradInput_frame)(
          THTensor *gradInput,
          THTensor *gradOutput,
          THTensor *weight,
          THTensor *fgradInput,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH)
{
  THTensor *gradOutput2d = THTensor_(newWithStorage2d)(
    gradOutput->storage, gradOutput->storageOffset,
    gradOutput->size[0], -1,
    gradOutput->size[1]*gradOutput->size[2]*gradOutput->size[3], -1
  );

  THTensor_(addmm)(fgradInput, 0, fgradInput, 1, weight, gradOutput2d);
  THTensor_(free)(gradOutput2d);

  THTensor_(zero)(gradInput);

  THNN_(unfolded_acc_vol)(
    fgradInput, gradInput,
    kT, kW, kH,
    dT, dW, dH,
    pT, pW, pH,
    gradInput->size[0], gradInput->size[1], gradInput->size[3], gradInput->size[2],
    gradOutput->size[1], gradOutput->size[3], gradOutput->size[2]
  );
}

void THNN_(VolumetricConvolutionMM_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput,
          THTensor *fgradInput,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH)
{
  int nOutputPlane = (int)weight->size[0];

  THNN_(VolumetricConvolutionMM_shapeCheck)(
        state, input, gradOutput, weight, NULL,
        kT, kW, kH, dT, dW, dH, pT, pW, pH);
  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  int freeWeight = THNN_(view_weight)(&weight);

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(resizeAs)(fgradInput, finput);
  // depending on the BLAS library, fgradInput (result tensor) might
  // be left uninitialized on zero alpha, which might lead to weird behavior
  // hence, to be safe, zero it
  THTensor_(zero)(fgradInput);
  THTensor *tweight = THTensor_(new)();
  THTensor_(transpose)(tweight, weight, 0, 1);

  if (input->nDimension == 4)
  {
    THNN_(VolumetricConvolutionMM_updateGradInput_frame)(
      gradInput, gradOutput, tweight, fgradInput,
      kT, kW, kH,
      dT, dW, dH,
      pT, pW, pH
    );
  }
  else
  {
    long T = input->size[0];
    long t;

//#pragma omp parallel for private(t)
    for (t = 0; t < T; t++)
    {
      THTensor *gradInput_t = THTensor_(newSelect)(gradInput, 0, t);
      THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
      THTensor *fgradInput_t = THTensor_(newSelect)(fgradInput, 0, t);

      THNN_(VolumetricConvolutionMM_updateGradInput_frame)(
        gradInput_t, gradOutput_t, tweight, fgradInput_t,
        kT, kW, kH,
        dT, dW, dH,
        pT, pW, pH
      );

      THTensor_(free)(gradInput_t);
      THTensor_(free)(gradOutput_t);
      THTensor_(free)(fgradInput_t);
    }
  }

  THTensor_(free)(tweight);
  THTensor_(free)(input);
  THTensor_(free)(gradOutput);
  if (freeWeight)
    THTensor_(free)(weight);
}

static void THNN_(VolumetricConvolutionMM_accGradParameters_frame)(
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,
          real scale)
{
  long i;
  THTensor *gradOutput2d = THTensor_(newWithStorage2d)(
    gradOutput->storage, gradOutput->storageOffset,
    gradOutput->size[0], -1,
    gradOutput->size[1]*gradOutput->size[2]*gradOutput->size[3], -1
  );

  THTensor *tfinput = THTensor_(new)();
  THTensor_(transpose)(tfinput, finput, 0, 1);
  THTensor_(addmm)(gradWeight, 1, gradWeight, scale, gradOutput2d, tfinput);
  THTensor_(free)(tfinput);

  if (gradBias) {
    for (i = 0; i < gradBias->size[0]; i++)
    {
      long k;
      real sum = 0;
      real *data = gradOutput2d->storage->data + gradOutput2d->storageOffset + i*gradOutput2d->stride[0];
      for (k = 0; k < gradOutput2d->size[1]; k++)
        sum += data[k];

      (gradBias->storage->data + gradBias->storageOffset)[i] += scale * sum;
    }
  }

  THTensor_(free)(gradOutput2d);
}

void THNN_(VolumetricConvolutionMM_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int pT, int pW, int pH,
          accreal scale_)
{
  real scale = TH_CONVERT_ACCREAL_TO_REAL(scale_);
  int freeWeight;
  int nOutputPlane = (int)gradWeight->size[0];

  THNN_(VolumetricConvolutionMM_shapeCheck)(
        state, input, gradOutput, gradWeight, gradBias,
        kT, kW, kH, dT, dW, dH, pT, pW, pH);
  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  freeWeight = THNN_(view_weight)(&gradWeight);

  if (input->nDimension == 4)   // non-batch mode
  {
    THNN_(VolumetricConvolutionMM_accGradParameters_frame)(gradOutput, gradWeight, gradBias, finput, scale);
  }
  else  // batch mode
  {
    long T = input->size[0];
    long t;

    for (t = 0; t < T; t++)
    {
      THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
      THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);

      THNN_(VolumetricConvolutionMM_accGradParameters_frame)(gradOutput_t, gradWeight, gradBias, finput_t, scale);

      THTensor_(free)(gradOutput_t);
      THTensor_(free)(finput_t);
    }
  }

  THTensor_(free)(input);
  THTensor_(free)(gradOutput);
  if (freeWeight)
    THTensor_(free)(gradWeight);
}

#endif
