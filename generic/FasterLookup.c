#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/FasterLookup.c"
#else

// add two vectors
static inline void nn_(FasterLookup_addVec)(
  real *res, real alpha, real *vec, int dim) {
  int i;
  int m = dim - 3;
  for (i = 0; i < m; i += 4) {
    res[i] += alpha * vec[i];
    res[i+1] += alpha * vec[i+1];
    res[i+2] += alpha * vec[i+2];
    res[i+3] += alpha * vec[i+3];
  }
  for ( ; i < dim; ++i)
    res[i] += alpha * vec[i];
}

// check if input goes outside allowed indicies
static int nn_(FasterLookup_boundError)(
  int n_inputs, int max_index, int* input) {
  int i; int idx; int err = 0;

  for(i=0; i<n_inputs; i++){
      idx = *input++;
      err = err || idx < 1 || idx > max_index;
  }

  return err;
}

// accumulate into (grad)weights
static void nn_(FasterLookup_acc)(THTensor *tWeight, real scale,
  THIntTensor *tInput, THTensor *tGradOutput, THIntTensor *tCount,
  int concUpdates){

  // make sure input, gradOutput are contiguous
  tInput = THIntTensor_newContiguous(tInput);
  tGradOutput = THTensor_(newContiguous)(tGradOutput);

  real * weight = THTensor_(data)(tWeight);
  int * input = THIntTensor_data(tInput);
  real * gradOutput = THTensor_(data)(tGradOutput);
  int * count = (tCount) ? (THIntTensor_data(tCount)) : (NULL);

  // update
  int n_inputs = THIntTensor_nElement(tInput);
  int dim = tWeight->size[1];
  int i;
  int idx;

  if (concUpdates) { // with OMP, concurrent updates, might drop some updates
    #pragma omp parallel for private(i, idx)
    for(i=0; i<n_inputs; i++){
      idx = input[i] - 1;
      real s = (count) ? (scale / (real)count[idx]) : scale;
      real *w = weight + dim * idx;
      nn_(FasterLookup_addVec)(w, s, gradOutput + dim * i, dim);
    }
  } else { // without OMP
    for(i=0; i<n_inputs; i++){
      idx = input[i] - 1;
      real s = (count) ? (scale / (real)count[idx]) : scale;
      real *w = weight + dim * idx;
      nn_(FasterLookup_addVec)(w, s, gradOutput + dim * i, dim);
    }
  }

  THIntTensor_free(tInput);
  THIntTensor_free(tGradOutput);
}

// count frequency of each index
static void nn_(FasterLookup_incrementCount)(
  THIntTensor *tInput, THIntTensor *tCount, int reset) {
  // make sure input is contiguous
  tInput = THIntTensor_newContiguous(tInput);
  int * input = THIntTensor_data(tInput);
  int * count = THIntTensor_data(tCount);

  int n_inputs = THIntTensor_nElement(tInput);
  int i;
  count -= 1; // this is lua everything starts at 1
  int * cur_input = input;
  // set to 0 seen indices if necessary
  if (reset) { for(i=0; i<n_inputs; i++){ count[*cur_input++] = 0; } }
  // count seen indices
  cur_input = input;
  for(i=0; i<n_inputs; i++){ count[*cur_input++]++; }

  THIntTensor_free(tInput);
 }

int nn_(FasterLookup_updateOutput)(lua_State *L) {
  THIntTensor *tInput = luaT_checkudata(L, 2, "torch.IntTensor");
  THTensor * tWeight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  int skipBC = luaT_getfieldcheckboolean(L, 1, "skipBoundChecking");
  THTensor * tOutput = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  tInput = THIntTensor_newContiguous(tInput);  // make sure input is contiguous

  int dim = tWeight->size[1];
  if (tInput->nDimension == 1) { // resize output
    THTensor_(resize2d)(tOutput, tInput->size[0], dim);
  } else if (tInput->nDimension == 2) {
    THTensor_(resize3d)(tOutput, tInput->size[0], tInput->size[1], dim);
  } else {
    luaL_error(L, "input should have 1 or 2 dimensions");
  }

  int n_inputs = THIntTensor_nElement(tInput);
  int *input = THIntTensor_data(tInput);   // pointers
  real * weight = THTensor_(data)(tWeight);
  real * output = THTensor_(data)(tOutput);

  if (!skipBC) { // bound checking?
    int max_index = tWeight->size[0];
    int err = nn_(FasterLookup_boundError)(n_inputs, max_index, input);
    if (err) { luaL_error(L, "input contains an index out of bounds"); }
  }

  int i;
  size_t vec_size = dim*sizeof(real);
  weight -= dim; // this is lua everything starts at 1
  #pragma omp parallel for private(i)
  for(i=0; i<n_inputs; i++){
      memcpy(output + i*dim, weight + input[i]*dim, vec_size);
  }

  THIntTensor_free(tInput);
  return 1;
}

int nn_(FasterLookup_updateParameters)(lua_State *L){
  real lr = (real)luaL_checknumber(L, 2);
  THTensor * tWeight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor * tGradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
  THIntTensor * tCount = luaT_getfieldcheckudata(L, 1, "count", "torch.IntTensor");
  int scaleGradByFreq = luaT_getfieldcheckboolean(L, 1, "scaleGradByFreq");

  real * weight = THTensor_(data)(tWeight);
  real * gradWeight = THTensor_(data)(tGradWeight);
  int * count = THIntTensor_data(tCount);

  int i;
  int c;
  int n_indexes = tWeight->size[0];
  int dim = tWeight->size[1];
  #pragma omp parallel for private(i, c)
  for(i=0; i < n_indexes; i++){
    c = count[i];
    if (c > 0) { // each non zero count need add
      real scale = (scaleGradByFreq) ? (lr / ((real)c)) : (lr);
      real *w = weight + dim * i;
      real *gw = gradWeight + dim * i;
      nn_(FasterLookup_addVec)(w, -scale, gw, dim);
    }
  }

  return 0;
}

int nn_(FasterLookup_accGradParameters)(lua_State *L){
  THTensor * tInput = luaT_checkudata(L, 2, "torch.IntTensor");
  THTensor * tGradOutput = luaT_checkudata(L, 3, torch_Tensor);
  real scale = (real)luaL_checknumber(L, 4);
  THTensor * tGradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);

  // increment count
  THIntTensor * tCount = luaT_getfieldcheckudata(L, 1, "count", "torch.IntTensor");
  int * count = THIntTensor_data(tCount);
  nn_(FasterLookup_incrementCount)(tInput, tCount, 0);

  // increment grad weight
  int concUpdates = luaT_getfieldcheckboolean(L, 1, "concUpdates");
  nn_(FasterLookup_acc)(tGradWeight, scale, tInput, tGradOutput, NULL, concUpdates);

  return 0;
}

int nn_(FasterLookup_accUpdateGradParameters)(lua_State *L){
  THTensor * tInput = luaT_checkudata(L, 2, "torch.IntTensor");
  THTensor * tGradOutput = luaT_checkudata(L, 3, torch_Tensor);
  real lr = (real)luaL_checknumber(L, 4);
  THTensor * tWeight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);

  // reset and increment count
  int *count = NULL;
  int scaleGradByFreq = luaT_getfieldcheckboolean(L, 1, "scaleGradByFreq");
  if (scaleGradByFreq) {
    THIntTensor * tCount = luaT_getfieldcheckudata(L, 1, "count", "torch.IntTensor");
    count = THIntTensor_data(tCount);
    nn_(FasterLookup_incrementCount)(tInput, tCount, 1);
  }

  // increment weight
  int concUpdates = luaT_getfieldcheckboolean(L, 1, "concUpdates");
  nn_(FasterLookup_acc)(tWeight, -lr, tInput, tGradOutput, count, concUpdates);
  return 0;
}

static const struct luaL_Reg nn_(FasterLookup__) [] = {
  {"FasterLookup_updateOutput", nn_(FasterLookup_updateOutput)},
  {"FasterLookup_updateParameters", nn_(FasterLookup_updateParameters)},
  {"FasterLookup_accGradParameters", nn_(FasterLookup_accGradParameters)},
  {"FasterLookup_accUpdateGradParameters",nn_(FasterLookup_accUpdateGradParameters)},
  {NULL, NULL}
};

void nn_(FasterLookup_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(FasterLookup__), "nn");
  lua_pop(L,1);
}

#endif
