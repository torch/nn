local ffi = require 'ffi'

local THNN = {}

-- load libTHNN
local cpath = package.cpath
if ffi.os == 'OSX' then cpath = string.gsub(package.cpath, '%?%.so;', '?.dylib;') end
THNN.C = ffi.load(package.searchpath('libTHNN', cpath))

local generic_THNN_h = [[
TH_API void THNN_(Abs_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output);
TH_API void THNN_(Abs_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput);
          
TH_API void THNN_(AbsCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          real *output,
          bool sizeAverage);
TH_API void THNN_(AbsCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage);
]]

-- THGenerator struct declaration copied from torch7/lib/TH/THRandom.h
local base_declarations = [[
typedef void THNNState;

typedef struct {
  unsigned long the_initial_seed;
  int left;
  int seeded;
  unsigned long next;
  unsigned long state[624]; /* the array for the state vector 624 = _MERSENNE_STATE_N  */
  double normal_x;
  double normal_y;
  double normal_rho;
  int normal_is_valid;
} THGenerator;
]]

ffi.cdef(base_declarations)

-- expand macros, allow to use original lines from lib/THNN/generic/THNN.h
local preprocessed = string.gsub(generic_THNN_h, 'TH_API void THNN_%(([%a%d_]+)%)', 'void THNN_TYPE%1')

local replacements = 
{
  { ['TYPE'] = 'Double', ['real'] = 'double', ['THTensor'] = 'THDoubleTensor', ['THIndexTensor'] = 'THLongTensor' },
  { ['TYPE'] = 'Float',  ['real'] = 'float',  ['THTensor'] = 'THFloatTensor',  ['THIndexTensor'] = 'THLongTensor' }
}

for i=1,#replacements do
  local r = replacements[i]
  local s = preprocessed
  for k,v in pairs(r) do
    s = string.gsub(s, k, v)
  end
  ffi.cdef(s)
end

THNN.NULL = ffi.NULL or nil

function THNN.getState()
  return ffi.NULL or nil
end

function THNN.optionalTensor(t)
  return t and t:cdata() or THNN.NULL
end

local function extract_function_names(s)
  local t = {}
  for n in string.gmatch(s, 'TH_API void THNN_%(([%a%d_]+)%)') do
    t[#t+1] = n
  end
  return t
end

function THNN.bind(lib, base_names, type_name, state_getter)
  local ftable = {}
  local prefix = 'THNN_' .. type_name
  for i,n in ipairs(base_names) do
    -- use pcall since some libs might not support all functions (e.g. cunn)
    local ok,v = pcall(function() return lib[prefix .. n] end)
    if ok then
      ftable[n] = function(...) v(state_getter(), ...) end   -- implicitely add state
    else
      print('not found: ' .. prefix .. n .. v)
    end
  end
  return ftable
end

-- build function table
local function_names = extract_function_names(generic_THNN_h)

THNN.kernels = {}
THNN.kernels['torch.FloatTensor'] = THNN.bind(THNN.C, function_names, 'Float', THNN.getState)
THNN.kernels['torch.DoubleTensor'] = THNN.bind(THNN.C, function_names, 'Double', THNN.getState)

torch.getmetatable('torch.FloatTensor').THNN = THNN.kernels['torch.FloatTensor']
torch.getmetatable('torch.DoubleTensor').THNN = THNN.kernels['torch.DoubleTensor']

function THNN.runKernel(f, type, ...)
  local ftable = THNN.kernels[type]
  if not ftable then
    error('Unsupported tensor type: '..type)
  end
  local f = ftable[f]
  if not f then
    error(string.format("Function '%s' not found for tensor type '%s'.", f, type))
  end
  f(...)
end

return THNN
