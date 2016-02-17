#include "TH.h"
#include "luaT.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define nn_(NAME) TH_CONCAT_3(nn_, Real, NAME)


#include "generic/SpatialConvolution.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialFullConvolutionMap.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialConvolutionMap.c"
#include "THGenerateFloatTypes.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libnn(lua_State *L);

int luaopen_libnn(lua_State *L)
{
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setglobal(L, "nn");

  nn_FloatSpatialConvolution_init(L);
  nn_FloatSpatialFullConvolutionMap_init(L);
  nn_FloatSpatialConvolutionMap_init(L);

  nn_DoubleSpatialConvolution_init(L);
  nn_DoubleSpatialFullConvolutionMap_init(L);
  nn_DoubleSpatialConvolutionMap_init(L);

  return 1;
}
