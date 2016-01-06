------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
API management file for Torch7 'nn' package.

Authored: 2016-01-06 (jwilson)
Modified: 2016-01-06
--]]

---------------- External Dependencies
local ffi = require 'ffi'
paths.dofile('API.lua')

---------------- Preamble for C code
local preamble =
[[
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
ffi.cdef(preamble)

------------------------------------------------
--                                          THNN
------------------------------------------------
---- Initialize API for THNN
local config  =
{
  libname = 'THNN',           tag   = 'TH_API',
  pattern = '%(([%a%d_]+)%)', rtype = 'void',
}
local THNN = API(config)

---- Load header, process macros, & forward declare
local header  = paths.dofile('THNN_h.lua')
local prep    = string.format('%s %s_TEMP%%1', THNN.config.rtype, THNN.config.libname)
local macros  = 
{
  {TEMP='Double',real='double',THTensor='THDoubleTensor',THIndexTensor='THLongTensor'},
  {TEMP='Float', real='float', THTensor='THFloatTensor', THIndexTensor='THLongTensor'}
}
THNN:forward_declare(header, macros, prep)

---- Bind C-functions to Lua library
local cstuct = 'libTHNN'
THNN.kernels = {}
THNN.kernels['torch.FloatTensor']  = THNN:bind(cstuct, header, 'Float')
THNN.kernels['torch.DoubleTensor'] = THNN:bind(cstuct, header, 'Double')

---- Make duplicate assignment (makes dynamic dispatching easy)
torch.getmetatable('torch.FloatTensor').THNN  = THNN.kernels['torch.FloatTensor']
torch.getmetatable('torch.DoubleTensor').THNN = THNN.kernels['torch.DoubleTensor']

---- Additional methods
THNN.NULL = ffi.NULL or nil
function THNN.optionalTensor(t)
   return t and t:cdata() or THNN.NULL
end

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
