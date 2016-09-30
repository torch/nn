#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/RReLU.c"
#else

void THNN_(RReLU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *noise,
          real lower,
          real upper,
          bool train,
          bool inplace,
          bool channelwise,
          THGenerator *generator)
{
  if (channelwise && train)
  {
    long bs, ks;
    long nOutputPlane;
    {
      long input_ndim = THTensor_(nDimension)(input);
      switch (input_ndim)
      {
        case 1:
          bs = 1;
          ks = 1;
          break;
        case 2:
          bs = input->size[0];
          ks = 1;
          break;
        case 3:
          bs = 1;
          ks = input->size[1] * input->size[2];
          break;
        case 4:
          bs = input->size[0];
          ks = input->size[2] * input->size[3];
          break;
      }
      nOutputPlane = input->size[(input_ndim + 1) % 2];
    }
    // get default random generator
    if (inplace)
      THTensor_(resizeAs)(noise, input);
    else
      THTensor_(resize1d)(noise, nOutputPlane);

    real *output_data = NULL;
    real *input_data = THTensor_(data)(input);
    real *noise_data = THTensor_(data)(noise);
    if (!inplace)
    {
      THTensor_(resizeAs)(output, input);
      output_data = THTensor_(data)(output);
    }
    THTensor *channel_noise = THTensor_(newWithSize1d)(nOutputPlane);
    real *channel_noise_data = THTensor_(data)(channel_noise);

    THIndex_t i, j, k;
#pragma omp parallel for private(j)
    for (j = 0; j < nOutputPlane; ++j)
      channel_noise_data[j] = (real)THRandom_uniform(generator, lower, upper);
#pragma omp parallel for private(j,k)
    for (i = 0; i < bs; ++i)
    {
      real* n_input_data = input_data + i*nOutputPlane*ks;
      real* n_output_data = NULL;
      real* n_noise_data = NULL;
      if (inplace)
        n_noise_data = noise_data + i*nOutputPlane*ks;
      else
        n_output_data = output_data + i*nOutputPlane*ks;
      for (j = 0; j < nOutputPlane; ++j)
      {
        const real r = channel_noise_data[j];
        for (k = 0; k < ks; ++k)
          if (inplace)
            if (n_input_data[k] <= 0)
            {
              n_input_data[k] = r * n_input_data[k];
              n_noise_data[k] = r;
            }
            else
              n_noise_data[k] = 1;
          else
            n_output_data[k] = (n_input_data[k] > 0) ? n_input_data[k] : r * n_input_data[k];
        n_input_data += ks;
        if (inplace)
          n_noise_data += ks;
        else
          n_output_data += ks;
      }
    }
    if (inplace)
      THTensor_(set)(output, input);
    else
      THTensor_(set)(noise, channel_noise);
  }
  else
  {
    if (train)
    {
      // get default random generator
      THTensor_(resizeAs)(noise, input);
      if (inplace)
      {
        TH_TENSOR_APPLY2(real, input, real, noise,
          if (*input_data <= 0)
          {
            const real r = (real)THRandom_uniform(generator, lower, upper);
            *input_data = (*input_data) * r;
            *noise_data = r;
          }
          else
          {
            *noise_data = 1;
          }
        );
        THTensor_(set)(output, input);
      }
      else
      {
        THTensor_(resizeAs)(output, input);
        TH_TENSOR_APPLY3(real, input, real, output, real, noise,
          if (*input_data <= 0)
          {
            const real r = (real)THRandom_uniform(generator, lower, upper);
            *output_data = (*input_data) * r;
            *noise_data = r;
          }
          else
          {
            *output_data = *input_data;
            *noise_data = 1;
          }
        );
      }
    }
    else
    {
      const real negSlope = (lower + upper) / 2;
      if (inplace)
      {
        TH_TENSOR_APPLY(real, input,
          if (*input_data <= 0)
          {
            *input_data = *input_data * negSlope;
          }
        );
        THTensor_(set)(output, input);
      }
      else
      {
        THTensor_(resizeAs)(output, input);
        TH_TENSOR_APPLY2(real, input, real, output,
          const real r = (*input_data) <= 0 ? negSlope : 1;
          *output_data = *input_data * r;
        );
      }
    }
  }
}

void THNN_(RReLU_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *noise,
          real lower,
          real upper,
          bool train,
          bool inplace,
          bool channelwise)
{
  THNN_CHECK_NELEMENT(input, gradOutput);
  if (train && upper - lower > 1E-6)    // e.g. if upper == lower, RReLU behaves like LeakyReLU
  {
    if (channelwise && !inplace)
    {
      long bs, ks;
      long nOutputPlane;
      {
        long input_ndim = THTensor_(nDimension)(input);
        switch (input_ndim)
        {
          case 1:
            bs = 1;
            ks = 1;
            break;
          case 2:
            bs = input->size[0];
            ks = 1;
            break;
          case 3:
            bs = 1;
            ks = input->size[1] * input->size[2];
            break;
          case 4:
            bs = input->size[0];
            ks = input->size[2] * input->size[3];
            break;
        }
        nOutputPlane = input->size[(input_ndim + 1) % 2];
      }

      const real *input_data = THTensor_(data)(input);
      const real *gradOutput_data = THTensor_(data)(gradOutput);
      THTensor_(resizeAs)(gradInput, input);
      real *gradInput_data = THTensor_(data)(gradInput);
      const real *noise_data = THTensor_(data)(noise);

      THIndex_t i, j, k;
#pragma omp parallel for private(j,k)
      for (i = 0; i < bs; ++i)
      {
        const real *n_input_data = input_data + i*nOutputPlane*ks;
        const real *n_gradOutput_data = gradOutput_data + i*nOutputPlane*ks;
        real *n_gradInput_data = gradInput_data + i*nOutputPlane*ks;

        for (j = 0; j < nOutputPlane; ++j)
        {
          const real r = noise_data[j];
          for (k = 0; k < ks; ++k)
            if (n_input_data[k] > 0)
              n_gradInput_data[k] = n_gradOutput_data[k];
            else
              n_gradInput_data[k] = n_gradOutput_data[k] * r;
          n_input_data += ks;
          n_gradInput_data += ks;
          n_gradOutput_data += ks;
        }
      }
    }
    else
    {
      // multiply the gradient by the noise tensor
      if (inplace)
      {
        THTensor_(cmul)(gradOutput, gradOutput, noise);
        THTensor_(set)(gradInput, gradOutput);
      }
      else
      {
        THTensor_(resizeAs)(gradInput, input);
        THTensor_(cmul)(gradInput, gradOutput, noise);
      }
    }
  }
  else
  {
    // use constant factor for negative input values
    const real negSlope = (lower + upper) / 2;
    if (inplace)
    {
      TH_TENSOR_APPLY2(real, gradOutput, real, input,
        if (*input_data <= 0)
        {
          *gradOutput_data = (*gradOutput_data) * negSlope;
        }
      );
      THTensor_(set)(gradInput, gradOutput);
    }
    else
    {
      THTensor_(resizeAs)(gradInput, input);
      TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,
        *gradInput_data = (*input_data) <= 0 ? (*gradOutput_data) * negSlope : (*gradOutput_data);
      );
    }
  }
}

#endif
