#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/TemporalSubSampling.c"
#else

static inline void THNN_(TemporalSubSampling_shapeCheck)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          int kW,
          int dW,
          int *inputFrameSize) {
  
  THArgCheck(kW > 0, 6,
             "kernel size should be greater than zero, but got kW: %d", kW);
  THArgCheck(dW > 0, 7,
             "stride should be greater than zero, but got dW: %d", dW);

  THNN_ARGCHECK(input->nDimension == 2 || input->nDimension == 3, 2, input,
                  "2D or 3D (batch mode) tensor expected for input, but got: %s");
  
  int dimS = 0; // sequence dimension
  int dimF = 1; // feature dimension
  
  if (input->nDimension == 3) {
    ++dimS;
    ++dimF;
  }
  
  if (inputFrameSize != NULL) {
    THArgCheck( input->size[dimF] == *inputFrameSize, 2,
                "invalid input frame size.  Got: %d, Expected: %d",
                input->size[dimF], *inputFrameSize);
  }
  THArgCheck( input->size[dimS] >= kW, 2,
              "input sequence smaller than kernel size.  Got %d, Expected: %d",
              input->size[dimS], kW);

  long nInputFrame = input->size[dimS];
  long nOutputFrame = (nInputFrame - kW) / dW + 1;

  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, input->nDimension, dimS, nOutputFrame);
    if (inputFrameSize != NULL) {
      THNN_CHECK_DIM_SIZE(gradOutput, input->nDimension, dimF, *inputFrameSize);
    }
  }
}

void THNN_(TemporalSubSampling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          int kW,
          int dW,
          int inputFrameSize)
{
	THTensor *outputFrame, *inputWindow;
	long nInputFrame, nOutputFrame;
	long k;

	THNN_(TemporalSubSampling_shapeCheck)(state, input, NULL, kW, dW, &inputFrameSize);

	outputFrame = THTensor_(new)();
	inputWindow = THTensor_(new)();

	int dimS = 0; // sequence dimension
	int dimF = 1; // feature dimension

	if (input->nDimension == 3)
	{
		++dimS;
		++dimF;
	}

	nInputFrame = input->size[dimS];
	nOutputFrame = (nInputFrame - kW) / dW + 1;

	if (input->nDimension == 2)
	{
		THTensor_(resize2d)(output, nOutputFrame, inputFrameSize);

		for(k = 0; k < nOutputFrame; k++)
		{
			THTensor_(narrow)(inputWindow, input, 0, k*dW, kW);
			THTensor_(select)(outputFrame, output, 0, k);
			THTensor_(sum)(outputFrame, inputWindow, 0);
			THTensor_(cmul)(outputFrame, outputFrame, weight);
			THTensor_(cadd)(outputFrame, outputFrame, 1, bias);
		}
	} else
	{
		THTensor *inputSample = THTensor_(new)();
		THTensor *outputSample = THTensor_(new)();
    long nBatchFrame = input->size[0];
    THTensor_(resize3d)(output, nBatchFrame, nOutputFrame, inputFrameSize);
		for (long i = 0; i < nBatchFrame; ++i)
		{
			THTensor_(select)(inputSample, input, 0, i);
			THTensor_(select)(outputSample, output, 0, i);

			for (k = 0; k < nOutputFrame; ++k)
			{
				THTensor_(narrow)(inputWindow, inputSample, 0, k*dW, kW);
				THTensor_(select)(outputFrame, outputSample, 0, k);
				THTensor_(sum)(outputFrame, inputWindow, 0);
				THTensor_(cmul)(outputFrame, outputFrame, weight);
			}
		}
		THTensor_(free)(inputSample);
		THTensor_(free)(outputSample);
	}
	THTensor_(free)(outputFrame);
	THTensor_(free)(inputWindow);
}

void THNN_(TemporalSubSampling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          int kW,
          int dW)
{

  THTensor *gradOutputFrame;
  THTensor *gradInputWindow, *buffer, *kwunit;
  long k;

  THNN_(TemporalSubSampling_shapeCheck)(state, input, gradOutput, kW, dW, NULL);

  gradOutputFrame = THTensor_(new)();
  gradInputWindow = THTensor_(new)();
  buffer = THTensor_(new)();
  kwunit = THTensor_(newWithSize1d)(kW);

  THTensor_(fill)(kwunit, 1);
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);
  
  int dimS = 0; // sequence dimension
  int dimF = 1; // feature dimension
  
  if (input->nDimension == 3) {
    ++dimS;
    ++dimF;
  }
  
  long nOutputFrame = gradOutput->size[dimS];

  if (input->nDimension == 2)
  {
    for(k = 0; k < nOutputFrame; k++)
    {
      THTensor_(narrow)(gradInputWindow, gradInput, 0, k*dW, kW);
      THTensor_(select)(gradOutputFrame, gradOutput, 0, k);
      THTensor_(cmul)(buffer, weight, gradOutputFrame);
      THTensor_(addr)(gradInputWindow, 1, gradInputWindow, 1, kwunit, buffer);
    }
  } else
  {
    THTensor *inputSample = THTensor_(new)();
    THTensor *outputSample = THTensor_(new)();
    long nBatchFrame = input->size[0];
    
    for (long i = 0; i < nBatchFrame; ++i)
    {
      THTensor_(select)(gradOutputSample, gradOutput, 0, i);
			THTensor_(select)(gradInputSample, gradInput, 0, i);

			for (k = 0; k < nOutputFrame; ++k)
      {
				THTensor_(narrow)(gradInputWindow, gradInputSample, 0, k*dW, kW);
				THTensor_(select)(gradOutputFrame, gradOutputSample, 0, k);
				THTensor_(cmul)(buffer, weight, gradOutputFrame);
				THTensor_(addr)(gradInputWindow, 1, gradInputWindow, 1, kwunit, buffer);
			}
		}
		THTensor_(free)(gradOutputSample);
		THTensor_(free)(gradInputSample);
	}

  THTensor_(free)(gradOutputFrame);
  THTensor_(free)(gradInputWindow);
  THTensor_(free)(buffer);
  THTensor_(free)(kwunit);
}

void THNN_(TemporalSubSampling_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          int kW,
          int dW,
          real scale)
{
  THTensor *gradOutputFrame;
  THTensor *inputWindow, *buffer;
  long k;

  THNN_(TemporalSubSampling_shapeCheck)(state, input, gradOutput, kW, dW, NULL);
  gradOutputFrame = THTensor_(new)();
  inputWindow = THTensor_(new)();
  buffer = THTensor_(new)();
  
  int dimS = 0; // sequence dimension
  int dimF = 1; // feature dimension
  
  if (input->nDimension == 2) {
    ++dimS;
    ++dimF;
  }
  
  long nOutputFrame = gradOutput->size[dimS];

  if (input->nDimension == 2)
  {
    for(k = 0; k < nOutputFrame; k++)
    {
      THTensor_(narrow)(inputWindow, input, 0, k*dW, kW);
      THTensor_(select)(gradOutputFrame, gradOutput, 0, k);
      THTensor_(sum)(buffer, inputWindow, 0);
      THTensor_(addcmul)(gradWeight, gradWeight, scale, buffer, gradOutputFrame);
      THTensor_(cadd)(gradBias, gradBias, scale, gradOutputFrame);
    }
  } else
  {
    THTensor *inputSample = THTensor_(new)();
		THTensor *gradOutputSample = THTensor_(new)();
    long nBatchFrame = input->size[0];
    
    for (long i = 0; i < nBatchFrame; ++i)
    {
      THTensor_(select)(inputSample, input, 0, i);
			THTensor_(select)(gradOutputSample, gradOutput, 0, i);
      
      for (k = 0; k < nOutputFrame; ++k)
      {
        THTensor_(narrow)(inputWindow, inputSample, 0, k*dW, kW);
				THTensor_(select)(gradOutputFrame, gradOutputSample, 0, k);
				THTensor_(sum)(buffer, inputWindow, 0);
				THTensor_(addcmul)(gradWeight, gradWeight, scale, buffer, gradOutputFrame);
        THTensor_(cadd)(gradBias, gradBias, scale, gradOutputFrame);
      }
    }
    THTensor_(free)(inputSample);
    THTensor_(free)(gradOutputSample);
  }

  THTensor_(free)(gradOutputFrame);
  THTensor_(free)(inputWindow);
  THTensor_(free)(buffer);
}

#endif
