#include "TH.h"
#include "luaT.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define nn_(NAME) TH_CONCAT_3(nn_, Real, NAME)

#include "generic/Square.c"
#include "THGenerateFloatTypes.h"

#include "generic/Sqrt.c"
#include "THGenerateFloatTypes.h"

#include "generic/Tanh.c"
#include "THGenerateFloatTypes.h"

#include "generic/SoftShrink.c"
#include "THGenerateFloatTypes.h"

#include "generic/Threshold.c"
#include "THGenerateFloatTypes.h"

#include "generic/SparseLinear.c"
#include "THGenerateFloatTypes.h"

#include "generic/TemporalConvolution.c"
#include "THGenerateFloatTypes.h"

#include "generic/TemporalSubSampling.c"
#include "THGenerateFloatTypes.h"

#include "generic/TemporalMaxPooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialBatchNormalization.c"
#include "THGenerateFloatTypes.h"

#include "generic/unfold.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialConvolution.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialConvolutionLocal.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialFullConvolution.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialFullConvolutionMap.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialConvolutionMap.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialSubSampling.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialMaxUnpooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialFractionalMaxPooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricConvolution.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricConvolutionMM.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricFullConvolution.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricMaxPooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricMaxUnpooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricAveragePooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialUpSamplingNearest.c"
#include "THGenerateFloatTypes.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libnn(lua_State *L);

int luaopen_libnn(lua_State *L)
{
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setglobal(L, "nn");

  nn_FloatSqrt_init(L);
  nn_FloatSquare_init(L);
  nn_FloatTanh_init(L);
  nn_FloatSoftShrink_init(L);
  nn_FloatThreshold_init(L);
  nn_FloatSparseLinear_init(L);
  nn_FloatTemporalConvolution_init(L);
  nn_FloatTemporalSubSampling_init(L);
  nn_FloatTemporalMaxPooling_init(L);
  nn_FloatSpatialBatchNormalization_init(L);
  nn_FloatSpatialConvolution_init(L);
  nn_FloatSpatialConvolutionLocal_init(L);
  nn_FloatSpatialFullConvolution_init(L);
  nn_FloatSpatialFullConvolutionMap_init(L);
  nn_FloatSpatialConvolutionMap_init(L);
  nn_FloatSpatialSubSampling_init(L);
  nn_FloatSpatialMaxUnpooling_init(L);
  nn_FloatSpatialFractionalMaxPooling_init(L);
  nn_FloatVolumetricConvolution_init(L);
  nn_FloatVolumetricConvolutionMM_init(L);
  nn_FloatVolumetricFullConvolution_init(L);
  nn_FloatVolumetricMaxPooling_init(L);
  nn_FloatVolumetricMaxUnpooling_init(L);
  nn_FloatVolumetricAveragePooling_init(L);
  nn_FloatSpatialUpSamplingNearest_init(L);

  nn_DoubleSqrt_init(L);
  nn_DoubleSquare_init(L);
  nn_DoubleTanh_init(L);
  nn_DoubleSoftShrink_init(L);
  nn_DoubleThreshold_init(L);
  nn_DoubleSparseLinear_init(L);
  nn_DoubleTemporalConvolution_init(L);
  nn_DoubleTemporalSubSampling_init(L);
  nn_DoubleTemporalMaxPooling_init(L);
  nn_DoubleSpatialBatchNormalization_init(L);
  nn_DoubleSpatialMaxUnpooling_init(L);
  nn_DoubleSpatialConvolution_init(L);
  nn_DoubleSpatialConvolutionLocal_init(L);
  nn_DoubleSpatialFullConvolution_init(L);
  nn_DoubleSpatialFullConvolutionMap_init(L);
  nn_DoubleSpatialFullConvolution_init(L);
  nn_DoubleSpatialConvolutionMap_init(L);
  nn_DoubleSpatialSubSampling_init(L);
  nn_DoubleSpatialFractionalMaxPooling_init(L);
  nn_DoubleVolumetricConvolution_init(L);
  nn_DoubleVolumetricConvolutionMM_init(L);
  nn_DoubleVolumetricFullConvolution_init(L);
  nn_DoubleVolumetricMaxPooling_init(L);
  nn_DoubleVolumetricMaxUnpooling_init(L);
  nn_DoubleVolumetricAveragePooling_init(L);
  nn_DoubleSpatialUpSamplingNearest_init(L);

  return 1;
}
