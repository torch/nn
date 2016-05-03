#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/MultiLabelMarginCriterion.c"
#else

void THNN_(MultiLabelMarginCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          THTensor *isTarget,
          bool sizeAverage)
{
  real *input_data, *target_data, *isTarget_data;
  long nframe, dim;
  long t, d, dt, ddt;
  real sum;

  THArgCheck((input->nDimension == 1) || (input->nDimension == 2), 2, "vector or matrix expected");

  if (input->nDimension == 1)
  {
    nframe = 1;
    dim = input->size[0];
    THArgCheck((target->nDimension == 1) && (target->size[0] == dim), 3, "inconsistent target size");
  }
  else
  {
    nframe = input->size[0];
    dim = input->size[1];
    THArgCheck((target->nDimension == 2) && (target->size[0] == nframe) && (target->size[1] == dim), 3, "inconsistent target size");
  }

  THArgCheck(THTensor_(minall)(target) >= 0, 3, "target out of range");
  THArgCheck(THTensor_(maxall)(target) <= dim, 3, "target out of range");

  target = THTensor_(newContiguous)(target);
  input = THTensor_(newContiguous)(input);
  input_data = THTensor_(data)(input);
  target_data = THTensor_(data)(target);

  THTensor_(resizeAs)(isTarget, target);
  THTensor_(zero)(isTarget);
  isTarget_data = THTensor_(data)(isTarget);

  sum = 0;
  for (t = 0; t < nframe; t++)
  {
    for (ddt = 0; ddt < dim; ddt++)
    {
      long target_idx = (long)target_data[ddt]-1;
      if (target_idx < 0)
        break;
      isTarget_data[target_idx] = 1;
    }
    for (dt = 0; dt < dim; dt++)
    {
      long target_idx = (long)target_data[dt]-1;
      real input_target;
      if (target_idx < 0)
        break;

      input_target = input_data[target_idx];
      for (d = 0; d < dim; d++)
      {
        if (!isTarget_data[d])
        {
          real z = 1 - input_target + input_data[d];
          if (z > 0)
            sum += z;
        }
      }
    }
    input_data += dim;
    target_data += dim;
    isTarget_data += dim;
  }

  sum /= dim;
  if (sizeAverage)
    sum /= nframe;

  THTensor_(set1d)(output, 0, sum);

  THTensor_(free)(input);
  THTensor_(free)(target);
}

void THNN_(MultiLabelMarginCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          THTensor *isTarget,
          bool sizeAverage)
{
  real *input_data;
  real *gradInput_data;
  real *target_data;
  real *isTarget_data;
  long nframe, dim;
  long t, d, dt;
  real g;

  THArgCheck((input->nDimension == 1) || (input->nDimension == 2), 2, "vector or matrix expected");

  if (input->nDimension == 1)
  {
    nframe = 1;
    dim = input->size[0];
    THArgCheck((target->nDimension == 1) && (target->size[0] == dim), 3, "inconsistent target size");
    THArgCheck((isTarget->nDimension == 1) && (isTarget->size[0] == dim), 3, "inconsistent isTarget size");
  }
  else
  {
    nframe = input->size[0];
    dim = input->size[1];
    THArgCheck((target->nDimension == 2) && (target->size[0] == nframe) && (target->size[1] == dim), 3, "inconsistent target size");
    THArgCheck((isTarget->nDimension == 2) && (isTarget->size[0] == nframe) && (isTarget->size[1] == dim), 3, "inconsistent isTarget size");
  }

  THArgCheck(THTensor_(minall)(target) >= 0, 3, "target out of range");
  THArgCheck(THTensor_(maxall)(target) <= dim, 3, "target out of range");

  THArgCheck(THTensor_(minall)(isTarget) >= 0, 3, "isTarget out of range");
  THArgCheck(THTensor_(maxall)(isTarget) <= 1, 3, "isTarget out of range");

  target = THTensor_(newContiguous)(target);
  input = THTensor_(newContiguous)(input);
  isTarget = THTensor_(newContiguous)(isTarget);
  input_data = THTensor_(data)(input);
  target_data = THTensor_(data)(target);
  isTarget_data = THTensor_(data)(isTarget);

  g = sizeAverage ? ( 1./((real)(nframe*dim)) ) : ( 1./((real)dim) );

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);
  gradInput_data = THTensor_(data)(gradInput);

  for (t = 0; t < nframe; t++)
  {
    for (dt = 0; dt < dim; dt++)
    {
      long target_idx = (long)target_data[dt]-1;
      real input_target;
      if (target_idx < 0)
        break;

      input_target = input_data[target_idx];
      for (d = 0; d < dim; d++)
      {
        if (!isTarget_data[d])
        {
          real z = 1 - input_target + input_data[d];
          if (z > 0)
          {
            gradInput_data[target_idx] -= g;
            gradInput_data[d] += g;
          }
        }
      }
    }
    input_data += dim;
    target_data += dim;
    isTarget_data += dim;
    gradInput_data += dim;
  }

  THTensor_(free)(input);
  THTensor_(free)(target);
  THTensor_(free)(isTarget);
}

#endif
