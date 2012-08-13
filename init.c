#include "TH.h"
#include "luaT.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define nn_(NAME) TH_CONCAT_3(nn_, Real, NAME)

#include "generic/Square.c"
#include "THGenerateFloatTypes.h"

#include "generic/Sqrt.c"
#include "THGenerateFloatTypes.h"

#include "generic/HardTanh.c"
#include "THGenerateFloatTypes.h"

#include "generic/Exp.c"
#include "THGenerateFloatTypes.h"

#include "generic/LogSigmoid.c"
#include "THGenerateFloatTypes.h"

#include "generic/LogSoftMax.c"
#include "THGenerateFloatTypes.h"

#include "generic/Sigmoid.c"
#include "THGenerateFloatTypes.h"

#include "generic/SoftPlus.c"
#include "THGenerateFloatTypes.h"

#include "generic/Tanh.c"
#include "THGenerateFloatTypes.h"

#include "generic/Abs.c"
#include "THGenerateFloatTypes.h"

#include "generic/HardShrink.c"
#include "THGenerateFloatTypes.h"

#include "generic/SoftShrink.c"
#include "THGenerateFloatTypes.h"

#include "generic/Threshold.c"
#include "THGenerateFloatTypes.h"

#include "generic/SoftMax.c"
#include "THGenerateFloatTypes.h"

#include "generic/Max.c"
#include "THGenerateFloatTypes.h"

#include "generic/Min.c"
#include "THGenerateFloatTypes.h"

#include "generic/MSECriterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/AbsCriterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/SparseLinear.c"
#include "THGenerateFloatTypes.h"

#include "generic/TemporalConvolution.c"
#include "THGenerateFloatTypes.h"

#include "generic/TemporalSubSampling.c"
#include "THGenerateFloatTypes.h"

#include "generic/TemporalMaxPooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialConvolution.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialConvolutionMap.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialSubSampling.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialMaxPooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricConvolution.c"
#include "THGenerateFloatTypes.h"

#include "generic/MultiMarginCriterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/MultiLabelMarginCriterion.c"
#include "THGenerateFloatTypes.h"

DLL_EXPORT int luaopen_libnn(lua_State *L)
{
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setfield(L, LUA_GLOBALSINDEX, "nn");

  nn_FloatMin_init(L);
  nn_FloatMax_init(L);
  nn_FloatExp_init(L);
  nn_FloatSqrt_init(L);
  nn_FloatSquare_init(L);
  nn_FloatHardTanh_init(L);
  nn_FloatLogSoftMax_init(L);
  nn_FloatMSECriterion_init(L);
  nn_FloatAbsCriterion_init(L);
  nn_FloatLogSigmoid_init(L);
  nn_FloatSigmoid_init(L);
  nn_FloatSoftMax_init(L);
  nn_FloatSoftPlus_init(L);
  nn_FloatTanh_init(L);
  nn_FloatAbs_init(L);
  nn_FloatHardShrink_init(L);
  nn_FloatSoftShrink_init(L);
  nn_FloatThreshold_init(L);
  nn_FloatSparseLinear_init(L);
  nn_FloatTemporalConvolution_init(L);
  nn_FloatTemporalSubSampling_init(L);
  nn_FloatTemporalMaxPooling_init(L);
  nn_FloatSpatialConvolution_init(L);
  nn_FloatSpatialConvolutionMap_init(L);
  nn_FloatSpatialSubSampling_init(L);
  nn_FloatSpatialMaxPooling_init(L);
  nn_FloatVolumetricConvolution_init(L);
  nn_FloatMultiMarginCriterion_init(L);
  nn_FloatMultiLabelMarginCriterion_init(L);

  nn_DoubleMin_init(L);
  nn_DoubleMax_init(L);
  nn_DoubleExp_init(L);
  nn_DoubleSqrt_init(L);
  nn_DoubleSquare_init(L);
  nn_DoubleHardTanh_init(L);
  nn_DoubleLogSoftMax_init(L);
  nn_DoubleMSECriterion_init(L);
  nn_DoubleAbsCriterion_init(L);
  nn_DoubleLogSigmoid_init(L);
  nn_DoubleSigmoid_init(L);
  nn_DoubleSoftMax_init(L);
  nn_DoubleSoftPlus_init(L);
  nn_DoubleTanh_init(L);
  nn_DoubleAbs_init(L);
  nn_DoubleHardShrink_init(L);
  nn_DoubleSoftShrink_init(L);
  nn_DoubleThreshold_init(L);
  nn_DoubleSparseLinear_init(L);
  nn_DoubleTemporalConvolution_init(L);
  nn_DoubleTemporalSubSampling_init(L);
  nn_DoubleTemporalMaxPooling_init(L);
  nn_DoubleSpatialConvolution_init(L);
  nn_DoubleSpatialConvolutionMap_init(L);
  nn_DoubleSpatialSubSampling_init(L);
  nn_DoubleSpatialMaxPooling_init(L);
  nn_DoubleVolumetricConvolution_init(L);
  nn_DoubleMultiMarginCriterion_init(L);
  nn_DoubleMultiLabelMarginCriterion_init(L);

  return 1;
}
