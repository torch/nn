#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/IndexLinear.c"
#else

#ifdef _OPENMP
#include <omp.h>
#endif

/* Threshold used to trigger multithreading */
#ifndef THNN_SPARSE_OMP_THRESHOLD
#define THNN_SPARSE_OMP_THRESHOLD 100000
#endif

/* Threshold used to trigger BLAS axpy call */
#ifndef THNN_SPARSE_OUTDIM_THRESHOLD
#define THNN_SPARSE_OUTDIM_THRESHOLD 49
#endif

/* sign MACRO */
#ifndef THNN_INDEXLINEAR_SIGN
#define THNN_INDEXLINEAR_SIGN(a) ( ( (a) < 0 )  ?  -1   : ( (a) > 0 ) )
#endif

static bool THNN_(checkKeysValues)(THLongTensor* keys, THTensor* values)
{
  return THLongTensor_size(keys, 0) == THTensor_(nElement)(values)
                && THTensor_(nDimension)(values) == 1
                && THLongTensor_nDimension(keys) == 1;
}

void THNN_(IndexLinear_updateOutput)(
          THNNState *state,
          THLongTensor *keys,
          long keysOffset,
          THTensor *values,
          THLongTensor *sizes,
          THLongTensor *cumSumSizes,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *normalizedValues,
          int  train)
{
  /* Retrieve all the dimensions of the problem */
  long batchSize = THLongTensor_size(sizes, 0);
  long keysSize = THLongTensor_size(keys, 0);
  long outDim = THTensor_(size)(bias, 0);
  long woutDim = THTensor_(size)(weight, 1);
  int maxNormalize = woutDim - outDim;
  long* sizesData = sizes->storage->data + sizes->storageOffset;
  long* cumSumSizesData = cumSumSizes->storage->data + cumSumSizes->storageOffset;

  /* Define/resize the normalized values tensor if maxNormalize is  > 0 */
  real* normalizedValuesData = NULL;
  if (maxNormalize)
  {
    THTensor_(resize1d)(normalizedValues, keysSize);
    normalizedValuesData = normalizedValues->storage->data + normalizedValues->storageOffset;
  }

  /* Resize the output */
  THTensor_(resize2d)(output, batchSize, outDim);

  /* Access the storage data/strides */
  real* outputData = output->storage->data + output->storageOffset;
  real* valuesData = values->storage->data + values->storageOffset;
  real* weightData = weight->storage->data + weight->storageOffset;
  long weightStride0 = weight->stride[0];
  long weightStride1 = weight->stride[1];
  real* biasData = bias->storage->data + bias->storageOffset;
  long* keysData = keys->storage->data + keys->storageOffset;

  /* Make sure these inputs are contiguous to accelerate computations */
  THArgCheck(THTensor_(isContiguous)(output), 1, "keys vector must be contiguous");
  THArgCheck(THLongTensor_isContiguous(keys), 1, "keys vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(values), 2, "values vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(normalizedValues), 2, "normalizedValues vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(bias), 7, "bias vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(weight), 6, "weight matrix must be contiguous");
  THArgCheck(THNN_(checkKeysValues)(keys, values), 1, "Keys and values should have the same number of elements");
  long i,j,k;

  /* Separate cases: output dimension is == 1, or > 1
   * This allows for some optimizations. */
  if (outDim == 1)
  {
    THVector_(fill)(outputData, *biasData, batchSize);
    if (maxNormalize)
    {
      /* Parallelize on the batch itself */
      #pragma omp parallel for private(i,j) firstprivate(outDim, keysOffset, weightData, keysData, valuesData, outputData, cumSumSizesData, sizesData) schedule(static)  if(keysSize*outDim > THNN_SPARSE_OMP_THRESHOLD && batchSize > 1)
      for (j = 0; j < batchSize; j++)
      {
        real* loutputData = outputData + j;
        real val = 0;
        real absVal = 0;
        long offset = cumSumSizesData[j];

        for (i = 0; i < sizesData[j]; i++)
        {
          long woffset = weightStride0*(keysData[offset] + keysOffset);
          absVal = fabs(valuesData[offset]);
          if (train)
          {
            if (absVal > weightData[woffset])
            {
              weightData[woffset] = absVal;
              weightData[woffset+1] = 1/absVal;
            }

            /*
             * The following can be used to scale the size of the updates
             * depending on some rule, e.g. the frequency of a feature, ...
             * This is used at update time.
             * TODO: implement a smarter update scale.
             */
            weightData[woffset+2] = 1;
          }
          normalizedValuesData[offset] = (absVal > weightData[woffset] ? THNN_INDEXLINEAR_SIGN(valuesData[offset]):valuesData[offset]*weightData[woffset+1]) + weightData[woffset+3];
          val += normalizedValuesData[offset] * weightData[woffset+maxNormalize];
          offset++;
        }
        *loutputData += val;
      }
    }
    else
    {
      /* Parallelize on the batch itself */
      #pragma omp parallel for private(i,j) firstprivate(outDim, weightData, keysData, valuesData, outputData, cumSumSizesData, sizesData) schedule(static)  if(keysSize*outDim > THNN_SPARSE_OMP_THRESHOLD && batchSize > 1)
      for (j = 0; j < batchSize; j++)
      {
        long offset = cumSumSizesData[j];
        real* loutputData = outputData + j;
        real val = 0;

        for (i = 0; i < sizesData[j]; i++) {
          val += weightData[weightStride0*(keysData[offset] + keysOffset)] * valuesData[offset];
          offset++;
        }
        *loutputData += val;
      }
    }
  }
  else {
    #pragma omp parallel for private(i,j,k) firstprivate(outDim, weightData, keysData, valuesData, biasData, outputData, cumSumSizesData, sizesData) schedule(static)  if(keysSize*outDim > THNN_SPARSE_OMP_THRESHOLD && batchSize > 1)
    for (j = 0; j < batchSize; j++)
    {
      long offset = cumSumSizesData[j];
      real val = 0;
      real* loutputData = outputData + j*outDim;
      real* lweightData = weightData;
      memcpy(loutputData, biasData, outDim*sizeof(real));
      for (i = 0; i < sizesData[j]; i++) {
        real val;
        long woffset = weightStride0*(keysData[offset] + keysOffset);
        if (maxNormalize)
        {
          val = valuesData[offset];
          real absVal = fabs(val);
          if (train)
          {
            if (absVal > weightData[woffset])
            {
              weightData[woffset] = absVal;
              weightData[woffset+1] = 1/absVal;
            }

            /*
             * The following can be used to scale the size of the updates
             * depending on some rule, e.g. the frequency of a feature, ...
             * The commented section thereafter is just an example of what can be done:
             *
             *```
             * weightData[woffset+2] = weightData[woffset+2]==0?1:(weightData[woffset+2] / (weightData[woffset+2] + 1));
             * real alpha = 1;
             * real beta = 0.01;
             * real gamma = 1 - 0.000001;
             * real l = weightData[woffset+2]==0?1/gamma:(weightData[woffset+2] - beta) / (alpha - beta);
             * l = gamma*l;
             * weightData[woffset+2] = (alpha-beta)*l + beta;
             * ```
             *
             * TODO: implement a smarter update scale.
             */
            weightData[woffset+2] = 1;
          }

          /* Normalize + Clamp */
          val = (absVal > weightData[woffset] ? THNN_INDEXLINEAR_SIGN(val):val*weightData[woffset+1]) + weightData[woffset+3];
          normalizedValuesData[offset] = val;

          lweightData = weightData + woffset + maxNormalize*weightStride1;
        }
        else
        {
          val = valuesData[offset];
          lweightData = weightData + woffset;
        }
        if (outDim > THNN_SPARSE_OUTDIM_THRESHOLD)
        {
          THBlas_(axpy)(outDim, val, lweightData, weightStride1, loutputData, 1);
        }
        else
        {
          for (k=0; k < outDim; k++)
          {
            loutputData[k] += lweightData[k*weightStride1] * val;
          }
        }
        offset++;
      }
    }
  }
  return;
}

void THNN_(IndexLinear_updateParameters)(
          THNNState *state,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *weight,
          THTensor *bias,
          THLongTensor *runningKeys,
          long keysOffset,
          real weightDecay,
          real learningRate)
{
  /* Retrieve all the dimensions of the problem */
  long outDim = THTensor_(size)(bias, 0);
  long woutDim = THTensor_(size)(weight, 1);
  int maxNormalize = woutDim - outDim;
  long keysSize = THLongTensor_size(runningKeys, 0);

  /* Access the storage data/strides */
  real* gradWeightData = THTensor_(data)(gradWeight) + gradWeight->storageOffset;
  real* weightData = THTensor_(data)(weight) + weight->storageOffset;
  long weightStride0 = weight->stride[0];
  long weightStride1 = weight->stride[1];
  real* gradBiasData = THTensor_(data)(gradBias) + gradBias->storageOffset;
  real* biasData = THTensor_(data)(bias) + bias->storageOffset;
  long* keysData = THLongTensor_data(runningKeys) + runningKeys->storageOffset;

  /* Make sure these inputs are contiguous to accelerate computations */
  THArgCheck(THLongTensor_isContiguous(runningKeys), 1, "keys vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(bias), 6, "gradBias vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(weight), 5, "gradBias vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(gradBias), 4, "gradBias vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(gradWeight), 3, "gradWeight must be contiguous");

  int j,k;
  long offset = 0;

  /* Update the bias first */
  THVector_(add)(biasData, gradBiasData, -learningRate, outDim);

  /* Separate cases: output dimension is == 1, or > 1
   * This allows for some optimizations.
   * No multithreading here as this could
   * corrupt the results (hogwild style) */
  if (outDim == 1)
  {
    if (maxNormalize)
    {
      if (weightDecay)
      {
        for (j = 0; j < keysSize; j++)
        {
          long woffset = weightStride0*(keysData[j] + keysOffset) + maxNormalize;
          real lr = learningRate*weightData[woffset-2];
          weightData[woffset-1] -= weightData[woffset]*gradWeightData[2*j]*lr;
          weightData[woffset] -= gradWeightData[2*j+1]*lr - weightDecay * weightData[woffset-2] * weightData[woffset];
        }
      }
      else
      {
        for (j = 0; j < keysSize; j++)
        {
          long woffset = weightStride0*(keysData[j] + keysOffset) + maxNormalize;
          real lr = learningRate*weightData[woffset-2];
          weightData[woffset-1] -= weightData[woffset]*gradWeightData[2*j]*lr;
          weightData[woffset] -= gradWeightData[2*j+1]*lr;
        }
      }
    }
    else
    {
      if (weightDecay)
      {
        for (j = 0; j < keysSize; j++)
        {
          long woffset = weightStride0*(keysData[j] + keysOffset);
          weightData[woffset] -= gradWeightData[j]*learningRate + weightDecay * weightData[woffset];
        }
      }
      else
      {
        for (j = 0; j < keysSize; j++)
        {
          weightData[weightStride0*(keysData[j] + keysOffset)] -= gradWeightData[j]*learningRate;
        }
      }
    }
  }
  else
  {
    for (j = 0; j < keysSize; j++)
    {
      real lr = learningRate;
      real wd = weightDecay;
      real* lweightData;
      long woffset = weightStride0*(keysData[j] + keysOffset);
      real* lgradWeightData = gradWeightData + j*outDim;
      if (maxNormalize)
      {
        lgradWeightData += j*outDim;
        lweightData = weightData + woffset + maxNormalize - 2;
        lr = lr*lweightData[0];
        wd = weightDecay*lweightData[0];
        lweightData++;
        for (k=0; k < outDim; k++)
        {
            lweightData[0] -= lgradWeightData[k]*lweightData[k+1]*lr;
        }
        lweightData++;
        lgradWeightData += outDim;
      }
      else
      {
        lweightData = weightData + woffset;
      }

      /* We do sparse weight decay.
       * We think it makes more sense. */
      if (weightDecay)
      {
        for (k=0; k < outDim; k++)
        {
            lweightData[k] -= lweightData[k]*wd;
        }
      }

      if (outDim > THNN_SPARSE_OUTDIM_THRESHOLD)
      {
        THBlas_(axpy)(outDim, -lr, lgradWeightData, 1, lweightData, 1);
      }
      else
      {
        for (k=0; k < outDim; k++)
        {
          lweightData[k] -= lgradWeightData[k]*lr;
        }
      }
    }
  }
}


void THNN_(IndexLinear_accUpdateGradParameters)(
          THNNState *state,
          THLongTensor *keys,
          long keysOffset,
          THTensor *values,
          THLongTensor *sizes,
          THTensor *gradOutput,
          THTensor *weight,
          THTensor *bias,
          real weightDecay,
          real scale)
{
  /* Retrieve all the dimensions of the problem */
  long batchSize = THLongTensor_size(sizes, 0);
  long keysSize = THLongTensor_size(keys, 0);
  long outDim = THTensor_(size)(bias, 0);
  long woutDim = THTensor_(size)(weight, 1);
  int maxNormalize = woutDim - outDim;
  THArgCheck(THNN_(checkKeysValues)(keys, values), 1, "Keys and values should have the same number of elements");

  /* Access the storage data/strides */
  real* gradOutputData = THTensor_(data)(gradOutput) + gradOutput->storageOffset;
  real* valuesData =THTensor_(data)(values) + values->storageOffset;
  real* weightData = THTensor_(data)(weight) + weight->storageOffset;
  real* biasData = THTensor_(data)(bias) + bias->storageOffset;
  long weightStride0 = weight->stride[0];
  long weightStride1 = weight->stride[1];
  long biasStride = bias->stride[0];
  long* keysData = THLongTensor_data(keys) + keys->storageOffset;
  long* sizesData = THLongTensor_data(sizes) + sizes->storageOffset;

  /* Make sure these inputs are contiguous to accelerate computations */
  THArgCheck(THLongTensor_isContiguous(keys), 1, "keys vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(values), 2, "values vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(values), 6, "weight matrix must be contiguous");
  THArgCheck(THTensor_(isContiguous)(gradOutput), 5, "gradOutput vector must be contiguous");

  int i,j,k;

  /* Separate cases: output dimension is == 1, or > 1
   * This allows for some optimizations.
   * No multithreading here as this could
   * corrupt the results (hogwild style) */
  if (outDim == 1)
  {
    if (maxNormalize)
    {
        long offset = 0;
        for (j = 0; j < batchSize; j++)
        {
          real* lgradOutputData = gradOutputData + j;
          *biasData -= *lgradOutputData * scale;
          real val = *lgradOutputData * scale;
          real* lweightData = weightData;
          for (i = 0; i < sizesData[j]; i++) {
            long idx = weightStride0*(keysData[offset] + keysOffset) + maxNormalize;
            weightData[idx-1] -= weightData[idx]*val*weightData[idx-2];
            weightData[idx] -= (val*valuesData[offset] - weightDecay * weightData[idx])*weightData[idx-2];
            offset++;
          }
        }

        offset = 0;
        for (j = 0; j < batchSize; j++)
        {
          real* lweightData = weightData;
          for (i = 0; i < sizesData[j]; i++) {
            long idx = weightStride0*(keysData[offset] + keysOffset) + maxNormalize;
            weightData[idx-2] = 0;
            offset++;
          }
        }
    }
    else
    {
      if (weightDecay)
      {
        long offset = 0;
        for (j = 0; j < batchSize; j++)
        {
          real* lgradOutputData = gradOutputData + j;
          *biasData -= *lgradOutputData * scale;
          real val = *lgradOutputData * scale;
          real* lweightData = weightData;
          for (i = 0; i < sizesData[j]; i++) {
            long idx = weightStride0*(keysData[offset] + keysOffset);
            weightData[idx] -= val * valuesData[offset] + weightData[idx] * weightDecay;
            offset++;
          }
        }
      }
      else
      {
        long offset = 0;
        for (j = 0; j < batchSize; j++)
        {
          real val = gradOutputData[j] * scale;
          for (i = 0; i < sizesData[j]; i++) {
            weightData[(keysData[offset] + keysOffset)*weightStride0] -= val * valuesData[offset];
            offset++;
          }
          *biasData -= val;
        }
      }
    }
  }
  else {
    long offset = 0;
    for (j = 0; j < batchSize; j++)
    {
      real val = 0;
      real* lgradOutputData = gradOutputData + j*outDim;
      real* lweightData = weightData;
      THVector_(add)(biasData, lgradOutputData, -scale, outDim);
      for (i = 0; i < sizesData[j]; i++) {
        real val = valuesData[offset] * scale;
        real wd = weightDecay;

        // Max normalize case
        if (maxNormalize)
        {
          lweightData = weightData + weightStride0*(keysData[offset] + keysOffset) + (maxNormalize-2)*weightStride1;
          val *= lweightData[0];
          wd *= lweightData[0];
          for (k=0; k < outDim; k++)
          {
            lweightData[1] -= lweightData[k+2]*scale*lgradOutputData[k]*lweightData[0];
          }
          lweightData += 2;
        }
        else
        {
          lweightData = weightData + weightStride0*(keysData[offset] + keysOffset);
        }

        /* We do sparse weight decay.
         * We think it makes more sense. */
        if (weightDecay)
        {
          if (outDim > THNN_SPARSE_OUTDIM_THRESHOLD)
          {
            THBlas_(axpy)(outDim, -wd, lweightData, weightStride1, lweightData, weightStride1);
          }
          else
          {
            for (k=0; k < outDim; k++)
            {
              lweightData[k*weightStride1] -= wd * lweightData[k*weightStride1];
            }
          }
        }

        if (outDim > THNN_SPARSE_OUTDIM_THRESHOLD)
        {
          THBlas_(axpy)(outDim, -val, lgradOutputData, 1, lweightData, weightStride1);
        }
        else
        {
          for (k=0; k < outDim; k++)
          {
            lweightData[k*weightStride1] -= val * lgradOutputData[k];
          }
        }
        offset++;
      }
    }

    /* Max Normalize case:
     * Reset the smart update scaling if
     * one does it batch-wise.
     * TODO: Decide what to do with that piece of code.
     * NB: If the code belowe is uncommented, so should the commented
     * code in IndexLinear:zeroGradParameters() */

    /*
    if (maxNormalize)
    {
      offset = 0;
      for (j = 0; j < batchSize; j++)
      {
        real* lweightData = weightData;
        for (i = 0; i < sizesData[j]; i++)
        {
          real val = valuesData[offset] * scale;
          real wd = weightDecay;

          lweightData = weightData + weightStride0*(keysData[offset] + keysOffset) + (maxNormalize-2)*weightStride1;
          lweightData[0] = 0;
          offset++;
        }
      }
    }
    */
  }
  return;
}

void THNN_(IndexLinear_accGradParameters)(
          THNNState *state,
          THLongTensor *keys,
          long keysOffset,
          THTensor *values,
          THLongTensor *sizes,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *weight,
          THTensor *bias,
          THTensor *valuesBuffer,
          real weightDecay,
          real scale)
{
  /* Retrieve all the dimensions of the problem */
  long batchSize = THLongTensor_size(sizes, 0);
  long keysSize = THLongTensor_size(keys, 0);
  long outDim = THTensor_(size)(bias, 0);
  long woutDim = THTensor_(size)(weight, 1);
  long maxNormalize = (woutDim - outDim) > 0 ?1:0;
  THArgCheck(THNN_(checkKeysValues)(keys, values), 1, "Keys and values should have the same number of elements");
  long* sizesData = sizes->storage->data + sizes->storageOffset;

  /* COmpute the cumulative sizes */
  THLongTensor* cumSizes = THLongTensor_new();
  THLongTensor_cumsum(cumSizes, sizes, 0);
  long* cumSizesData = cumSizes->storage->data +cumSizes->storageOffset;

  /* Resize the gradWeight buffer to keep it dense.
   * That speeds up updates A LOT assuming random mem access. */
  THTensor_(resize2d)(gradWeight, keysSize, outDim * (maxNormalize>0?2:1));

  /* Access the storage data/strides */
  real* gradOutputData = THTensor_(data)(gradOutput) + gradOutput->storageOffset;
  real* valuesData =THTensor_(data)(values) + values->storageOffset;
  real* gradWeightData = THTensor_(data)(gradWeight) + gradWeight->storageOffset;
  real* weightData = THTensor_(data)(weight) + weight->storageOffset;
  real* gradBiasData = THTensor_(data)(gradBias) + gradBias->storageOffset;
  long gradWeightStride0 = gradWeight->stride[0];
  long gradWeightStride1 = gradWeight->stride[1];
  long weightStride0 = weight->stride[0];
  long weightStride1 = weight->stride[1];
  long* keysData = THLongTensor_data(keys) + keys->storageOffset;

  /* Make sure these inputs are contiguous to accelerate computations */
  THArgCheck(THLongTensor_isContiguous(keys), 2, "keys vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(values), 3, "values vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(gradBias), 8, "gradBias vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(gradOutput), 6, "gradOutput vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(gradWeight), 7, "gradWeight must be contiguous");
  THArgCheck(THTensor_(isContiguous)(valuesBuffer), 11, "valuesBuffer must be contiguous");

  int i,j,k;

  /* Separate cases: output dimension is == 1, or > 1
   * This allows for some optimizations.
   * No multithreading here as this could
   * corrupt the results (hogwild style) */
  if (outDim == 1)
  {
    for (j = 0; j < batchSize; j++)
    {
      long offset = j==0?0:cumSizesData[j-1];
      real val = gradOutputData[j] * scale;
      real* lgradWeightData = gradWeightData + offset;
      real* lvaluesData = valuesData + offset;
      long end = sizesData[j];

      if (maxNormalize)
      {
        lgradWeightData += offset;
        i = 0;
        for(;i < end; i++)
        {
          lgradWeightData[2*i] = val;
          lgradWeightData[2*i+1] = val * lvaluesData[i];
        }
      }
      else
      {
        i = 0;
        for(;i < end-4; i += 4)
        {
          lgradWeightData[i] = val * lvaluesData[i];
          lgradWeightData[i+1] = val * lvaluesData[i+1];
          lgradWeightData[i+2] = val * lvaluesData[i+2];
          lgradWeightData[i+3] = val * lvaluesData[i+3];
        }

        for(; i < end; i++) {
          lgradWeightData[i] = val * lvaluesData[i];
        }
      }
      *gradBiasData += val;
      offset += end;
    }
  }
  else {
    for (j = 0; j < batchSize; j++)
    {
      long offset = j==0?0:cumSizesData[j-1];
      real val = 0;
      real* lgradOutputData = gradOutputData + j*outDim;
      real* lgradWeightData = gradWeightData;
      real* lweightData = weightData;
      THVector_(add)(gradBiasData, lgradOutputData, scale, outDim);
      for (i = 0; i < sizesData[j]; i++)
      {
        real val = valuesData[offset] * scale;
        lgradWeightData = gradWeightData + offset*outDim;
        if (maxNormalize)
        {
          lgradWeightData += offset*outDim;
          k = 0;
          for(;k < outDim-4; k += 4)
          {
            lgradWeightData[k] = lgradOutputData[k]*scale;
            lgradWeightData[k+1] = lgradOutputData[k+1]*scale;
            lgradWeightData[k+2] = lgradOutputData[k+2]*scale;
            lgradWeightData[k+3] = lgradOutputData[k+3]*scale;
          }

          for(; k < outDim; k++) {
            lgradWeightData[k] = lgradOutputData[k]*scale;
          }
          lgradWeightData += outDim;
        }
        k = 0;
        for(;k < outDim-4; k += 4)
        {
          lgradWeightData[k] = val * lgradOutputData[k];
          lgradWeightData[k+1] = val * lgradOutputData[k+1];
          lgradWeightData[k+2] = val * lgradOutputData[k+2];
          lgradWeightData[k+3] = val * lgradOutputData[k+3];
        }

        for(; k < outDim; k++) {
          lgradWeightData[k] = val * lgradOutputData[k];
        }
        offset++;
      }
    }
  }
  THLongTensor_free(cumSizes);
  return;
}
#endif
