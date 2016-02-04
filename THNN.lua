
------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
* API management file for Torch7 'nn' package.
--]]

---------------- External Dependencies
local ffi = require 'ffi'
require('torchAPI')

------------------------------------------------
--                                          THNN
------------------------------------------------
---- Load header file
local header = paths.dofile('THNN_h.lua')

---- Initialize API for THNN
local config  =
{
  library = 'THNN',           tag   = 'TH_API',
  pattern = '%(([%a%d_]+)%)', rtype = 'void',
}
local THNN = torchAPI(config, header)


---- Bind C-functions to THNN library
local cstuct = 'libTHNN'
THNN.kernels = {}
THNN.kernels['torch.FloatTensor']  = THNN:bind(cstuct, header['forward'], 'Float')
THNN.kernels['torch.DoubleTensor'] = THNN:bind(cstuct, header['forward'], 'Double')

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
