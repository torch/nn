#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/WeightedSmoothL1Criterion.c"
#else

void THNN_(WeightedSmoothL1Criterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage, 
          THTensor *weights,           // [OPTIONAL] class weights
          THTensor *total_weight)
{
  weights = weights ? THTensor_(newContiguous)(weights) : NULL;
  real *total_weight_data = THTensor_(data)(total_weight);

  real total_weight_acc = 0;
  real sum = 0;

  if (weights) {
    TH_TENSOR_APPLY3(real, input, real, target, real, weights, 
      real z = fabs(*input_data - *target_data);
      sum += *weights_data * (z < 1 ? 0.5*z*z : z - 0.5);
      total_weight_acc += *weights_data;
    );
    *total_weight_data = total_weight_acc;
  }
  else {
    TH_TENSOR_APPLY2(real, input, real, target,
      real z = fabs(*input_data - *target_data);
      sum += z < 1 ? 0.5*z*z : z - 0.5;
      total_weight_acc += 1.0f;
    );
    *total_weight_data = total_weight_acc;
  }
 
  //if (sizeAverage)
  //  sum /= THTensor_(nElement)(input);
  if (sizeAverage && *total_weight_data > 1e-10)
    sum /= *total_weight_data;

  THTensor_(set1d)(output, 0, sum);
  if (weights)
    THTensor_(free)(weights);
}

void THNN_(WeightedSmoothL1Criterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage, 
          THTensor *weights,           // [OPTIONAL] class weights
          THTensor *total_weight)
{
  weights = weights ? THTensor_(newContiguous)(weights) : NULL;
  real *total_weight_data = THTensor_(data)(total_weight);

  //real norm = (sizeAverage ? 1./((real)THTensor_(nElement)(input)) : 1.);
  real normalize = sizeAverage ? *total_weight_data : 1.0f;

  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
    real x = *input_data - *target_data;
    if (x < -1.)
     *gradInput_data = - 1.0f; //- norm;
    else if (x > 1.)
     *gradInput_data = 1.0f; //norm;
    else
     *gradInput_data = x; //norm * x;
  );

  if (weights) {
    TH_TENSOR_APPLY2(real, gradInput, real, weights,
       *gradInput_data *= *weights_data / normalize; 
    );
    THTensor_(free)(weights);
  }
  else {
    TH_TENSOR_APPLY(real, gradInput,
       *gradInput_data /= normalize;
    );
  }
}

#endif
