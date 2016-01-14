#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/unfold.h"
#else


void nn_(unfolded_acc)(THTensor *finput, THTensor *input,
                       int kW, int kH, int dW, int dH, int padW, int padH,
                       int nInputPlane, int inputWidth, int inputHeight,
                       int outputWidth, int outputHeight);
                               
void nn_(unfolded_copy)(THTensor *finput, THTensor *input,
                        int kW, int kH, int dW, int dH, int padW, int padH,
                        int nInputPlane, int inputWidth, int inputHeight,
                        int outputWidth, int outputHeight);                             
#endif