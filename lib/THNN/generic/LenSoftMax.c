#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/LenSoftMax.c"
#else

void THNN_(LenSoftMax_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *len)
{
  if ((input->nDimension != 2) && (len->nDimension != 1))
  {
    THArgCheck(0, 2, "2D tensor expected for input, 1D tensor expected for len");
  }

  real *input_data, *output_data;
  THIndex_t *len_data;
  ptrdiff_t nframe = input->size[0], dim = input->size[1];
  ptrdiff_t t;

  input = THTensor_(newContiguous)(input);
  THTensor_(resizeAs)(output, input);

  input_data = THTensor_(data)(input);
  output_data = THTensor_(data)(output);
  len_data = THIndexTensor_(data)(len);

#pragma omp parallel for private(t)
  for (t = 0; t < nframe; t++)
  {
    real *input_ptr = input_data + t*dim;
    real *output_ptr = output_data + t*dim;

    real inputMax = -THInf;
    accreal sum;

    ptrdiff_t d, ld = (ptrdiff_t)len_data[t];
    for (d = 0; d < ld; d++)
    {
      if (input_ptr[d] >= inputMax) inputMax = input_ptr[d];
    }

    sum = 0;
    for (d = 0; d < ld; d++)
    {
      real z = exp(input_ptr[d] - inputMax);
      output_ptr[d] = z;
      sum += z;
    }
	for (d = ld; d < dim; d++)
    {
      output_ptr[d] = 0;
    }

    for (d = 0; d < ld; d++)
    {
      output_ptr[d] *= 1/sum;
    }
  }

  THTensor_(free)(input);
}

void THNN_(LenSoftMax_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output,
          THIndexTensor *len)
{
  THNN_CHECK_SHAPE(input, gradOutput);  

  if ((output->nDimension != 2) && (len->nDimension != 1))
  {
    THError("2D tensor expected for input, 1D tensor expected for len");
  }

  real *gradInput_data, *gradOutput_data, *output_data;
  THIndex_t *len_data;
  ptrdiff_t nframe = output->size[0], dim = output->size[1];
  ptrdiff_t t;

  gradOutput = THTensor_(newContiguous)(gradOutput);
  output = THTensor_(newContiguous)(output);

  THTensor_(resizeAs)(gradInput, output);
  gradInput_data = THTensor_(data)(gradInput);
  output_data = THTensor_(data)(output);
  gradOutput_data = THTensor_(data)(gradOutput);
  len_data = THIndexTensor_(data)(len);

#pragma omp parallel for private(t)
  for (t = 0; t < nframe; t++)
  {
    real *gradInput_ptr = gradInput_data + t*dim;
    real *output_ptr = output_data + t*dim;
    real *gradOutput_ptr = gradOutput_data + t*dim;

    ptrdiff_t d, ld = (ptrdiff_t)len_data[t];
    accreal sum = 0;
    for (d = 0; d < ld; d++)
      sum += (accreal)gradOutput_ptr[d] * output_ptr[d];

    for (d = 0; d < ld; d++)
      gradInput_ptr[d] = output_ptr[d] * (gradOutput_ptr[d] - sum);

    for (d = ld; d < dim; d++)
      gradInput_ptr[d] = 0;
  }

  THTensor_(free)(gradOutput);
  THTensor_(free)(output);
}

#endif