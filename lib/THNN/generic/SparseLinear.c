#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SparseLinear.c"
#else

#ifdef _OPENMP
#include <omp.h>
#endif

#define ROW_PTR2(t, r) (THTensor_(data)(t) + (r) * (t)->stride[0])
#define COL_PTR2(t, c) (THTensor_(data)(t) + (c) * (t)->stride[1])

static bool THNN_(checkInput)(THTensor* t)
{
  return t->nDimension == 3 && t->size[2] == 2;
}

static bool THNN_(checkSize2D)(THTensor* t, long size0, long size1)
{
  return t->nDimension == 2 && t->size[0] == size0 && t->size[1] == size1;
}

static bool THNN_(checkSize1D)(THTensor* t, long size0)
{
  return t->nDimension == 1 && t->size[0] == size0;
}

static void THNN_(set1d)(THTensor *t, long x0, real value) {
  THStorage_(set)(t->storage, t->storageOffset + x0*t->stride[0], value);
}
static real THNN_(get3d)(const THTensor *t, long x0, long x1, long x2) {
  return THStorage_(get)(t->storage, t->storageOffset +
                         x0*t->stride[0] + x1*t->stride[1] + x2*t->stride[2]);
}
static real THNN_(get2d)(const THTensor *t, long x0, long x1) {
  return THStorage_(get)(t->storage, t->storageOffset +
                         x0*t->stride[0] + x1*t->stride[1]);
}

void THNN_(SparseLinear_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *cudaBuffer,
          THTensor *shardBuffer)
{
  long h, i;
  long outDim = THTensor_(size)(weight, 0);
  long inDim = THTensor_(size)(weight, 1);

  THArgCheck(THNN_(checkInput)(input), 2, "input size must be batchsize x nnz x 2");
  THArgCheck(THTensor_(isContiguous)(output), 3, "output must be contiguous");
  THArgCheck(THNN_(checkSize1D)(bias, outDim), 5, "bias size wrong");

  long batchSize = THTensor_(size)(input, 0);
  long nnz = THTensor_(size)(input, 1);
  THTensor_(resize2d)(output, batchSize, outDim);

  // output = weight * input + bias
  THTensor_(zero)(output);
#pragma omp parallel for private(h, i) schedule(static) if (   \
  batchSize > 1 && batchSize * nnz * outDim > 10000)
  for (h = 0; h < batchSize; h++) {
    for (i = 0; i < nnz; i++) {
      real val = THNN_(get3d)(input, h, i, 1);
      if (val == 0) {
        continue;
      }

      long offset = (long)(THNN_(get3d)(input, h, i, 0)) - 1;
      if (offset >= 0 && offset < inDim) {
        THBlas_(axpy)(outDim,
                      val,
                      COL_PTR2(weight, offset), weight->stride[0],
                      ROW_PTR2(output, h), output->stride[1]);
      } else {
        THError("index out of bound. updateOutput: %d not between 1 and %d",
                offset + 1, inDim);
      }
    }
  }

  THTensor* output_row = THTensor_(new)();
  for (h = 0; h < batchSize; h++) {
    THTensor_(select)(output_row, output, 0, h);
    THTensor_(cadd)(output_row, bias, 1.0, output_row);
  }
  THTensor_(free)(output_row);
}

void THNN_(SparseLinear_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *weight,
          THTensor *bias,
          real weightDecay,
          real scale)
{
  long h, i;
  long outDim = THTensor_(size)(weight, 0);
  long inDim = THTensor_(size)(weight, 1);

  THArgCheck(THNN_(checkInput)(input), 2,
             "input size must be batchsize x nnz x 2");
  THArgCheck(THNN_(checkSize2D)(gradWeight, outDim, inDim), 4,
             "gradWeight size wrong");
  THArgCheck(THNN_(checkSize1D)(gradBias, outDim), 5,
             "gradBias size wrong");
  THArgCheck(THTensor_(isContiguous)(gradOutput), 1,
             "gradOutput must be contiguous");

  long batchSize = THTensor_(size)(input, 0);
  long nnz = THTensor_(size)(input, 1);
  THTensor_(resize2d)(gradOutput, batchSize, outDim);

  // gradWeight += gradOutput * input
#pragma omp parallel for private(h, i) schedule(static) if (\
  batchSize * nnz * outDim > 10000)
  for (i = 0; i < nnz; i++) {
    for (h = 0; h < batchSize; h++) {
      real val = scale * THNN_(get3d)(input, h, i, 1);
      if (val == 0) {
        continue;
      }

      long offset = (long)(THNN_(get3d)(input, h, i, 0)) - 1;
      if (offset >= 0 && offset < inDim) {
        THBlas_(axpy)(outDim,
                      val,
                      ROW_PTR2(gradOutput, h), gradOutput->stride[1],
                      COL_PTR2(gradWeight, offset), gradWeight->stride[0]);
      } else {
        THError(
          "index out of bound. accGradParameters: %d not between 1 and %d",
          offset + 1,
          inDim);
      }
    }
  }

  // gradBias += gradOutput
  THTensor* gradOutput_row = THTensor_(new)();
  for (h = 0; h < batchSize; h++) {
    THTensor_(select)(gradOutput_row, gradOutput, 0, h);
    THTensor_(cadd)(gradBias, gradBias, scale, gradOutput_row);
  }
  THTensor_(free)(gradOutput_row);

  if (weightDecay != 0) {
    THTensor_(cadd)(gradWeight, gradWeight, weightDecay, weight);
  }
}

void THNN_(SparseLinear_updateParameters)(
          THNNState *state,
          THTensor *weight,
          THTensor *bias,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *lastInput,
          real learningRate)
{
  long h, i;
  long outDim = weight->size[0];
  long inDim = weight->size[1];

  THArgCheck(THNN_(checkSize2D)(gradWeight, outDim, inDim), 4,
             "gradWeight size wrong");
  THArgCheck(THNN_(checkSize1D)(bias, outDim), 3, "bias size wrong");
  THArgCheck(THNN_(checkSize1D)(gradBias, outDim), 5, "gradBias size wrong");
  THArgCheck(THNN_(checkInput)(lastInput), 6,
             "input size must be batchsize x nnz x 2");


  long batchSize = THTensor_(size)(lastInput, 0);
  long nnz = THTensor_(size)(lastInput, 1);

  // collect unique offsets of non-0 val in input
  THTensor* offsets = THTensor_(newWithSize1d)(batchSize * nnz);
  long cnt = 0;
  for (h = 0; h < batchSize; h++) {
    for (i = 0; i < nnz; i++) {
      real val = THNN_(get3d)(lastInput, h, i, 1);
      if (val == 0 ) {
        continue;
      }
      long offset = (long)(THNN_(get3d)(lastInput, h, i, 0)) - 1;
      if (offset >= 0 && offset < inDim) {
        THNN_(set1d)(offsets, cnt++, offset);
      } else {
        THError(
          "index out of bound. updateParameters: %d not between 1 and %d",
          offset + 1,
          inDim);
      }
    }
  }
  THTensor_(resize1d)(offsets, cnt);

  THTensor* uniqueOffsets = THTensor_(new)();
  THLongTensor* ri = THLongTensor_new();
  THTensor_(sort)(uniqueOffsets, ri, offsets, 0, 0);
  THLongTensor_free(ri);
  THTensor_(free)(offsets);

  cnt = 1;
  real* uniqueOffsets_p = THTensor_(data)(uniqueOffsets);
  for (i = 1; i < THTensor_(size)(uniqueOffsets, 0); i++) {
    if (uniqueOffsets_p[i] != uniqueOffsets_p[i - 1]) {
      uniqueOffsets_p[cnt++] = uniqueOffsets_p[i];
    }
  }
  THTensor_(resize1d)(uniqueOffsets, cnt);

  // weight += -learningRate * gradWeight
  THTensor_(cadd)(bias, bias, -learningRate, gradBias);
#pragma omp parallel for private(i) schedule(static) if (cnt * outDim > 10000)
  for (i = 0; i < cnt; i++) {
    long offset = (long)uniqueOffsets_p[i];
    THBlas_(axpy)(outDim,
                  -learningRate,
                  COL_PTR2(gradWeight, offset), gradWeight->stride[0],
                  COL_PTR2(weight, offset), weight->stride[0]);
  }

  THTensor_(free)(uniqueOffsets);
}

void THNN_(SparseLinear_zeroGradParameters)(
          THNNState *state,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *lastInput)
{
  long h, i, j;

  long outDim = gradWeight->size[0];
  long inDim = gradWeight->size[1];

  THArgCheck(THNN_(checkSize1D)(gradBias, outDim), 3, "gradBias size wrong");
  THArgCheck(THNN_(checkInput)(lastInput), 4,
             "input size must be batchsize x nnz x 2");

  THTensor_(zero)(gradBias);

  long batchSize = THTensor_(size)(lastInput, 0);
  long nnz = THTensor_(size)(lastInput, 1);

#pragma omp parallel for private(h, i, j) schedule(static) if (   \
  batchSize > 1 && batchSize * nnz * outDim > 10000)
  for (h = 0; h < batchSize; h++) {
    for (i = 0; i < nnz; i++) {
      if (THNN_(get3d)(lastInput, h, i, 1) == 0 ) {
        continue;
      }

      long offset = (long)(THNN_(get3d)(lastInput, h, i, 0)) - 1;
      if (offset >= 0 && offset < inDim) {
        real* pGradWeight = COL_PTR2(gradWeight, offset);
        if (gradWeight->stride[0] == 1) {
          THVector_(fill)(pGradWeight, 0, outDim);
        } else {
          long stride = gradWeight->stride[0];
          for (j = 0; j < outDim; ++j) {
            pGradWeight[j * stride] = 0;
          }
        }
      } else {
        THError(
          "index out of bound. zeroGradParameters: %d not between 1 and %d",
          offset + 1,
          inDim);
      }
    }
  }
}

void THNN_(SparseLinear_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight)
{
  long h, i;
  long outDim = weight->size[0];
  long inDim = weight->size[1];

  THArgCheck(THNN_(checkInput)(input), 2,
             "input must be a batchSize x nnz x 2 tensor");
  THArgCheck(THTensor_(isContiguous)(gradInput), 4,
             "gradInput must be contiguous");
  THArgCheck(THTensor_(isContiguous)(gradOutput), 3,
             "gradOutput must be contiguous");

  long batchSize = THTensor_(size)(input, 0);
  long nnz = THTensor_(size)(input, 1);
  THTensor_(resize2d)(gradOutput, batchSize, outDim);
  THTensor_(resize3d)(gradInput, batchSize, nnz, 2);

#pragma omp parallel for private(h, i) schedule(static) if (    \
  batchSize > 1 && batchSize * nnz * outDim > 10000)
  for (h = 0; h < batchSize; h++) {
    for (i = 0; i < nnz; ++i) {
      long offset = (long)(THTensor_(get3d)(input, h, i, 0)) - 1;
      THTensor_(set3d)(gradInput, h, i, 0, offset + 1);

      if (offset >= 0 && offset < inDim) {
        real val = THBlas_(dot)(
          outDim,
          ROW_PTR2(gradOutput, h), gradOutput->stride[1],
          COL_PTR2(weight, offset), weight->stride[0]);
        THTensor_(set3d)(gradInput, h, i, 1, val);
      } else {
        THError(
          "index out of bound. updateGradInput: %d not between 1 and %d",
          offset + 1,
          inDim);
      }
    }
  }
}

#undef ROW_PTR2
#undef COL_PTR2

#endif
