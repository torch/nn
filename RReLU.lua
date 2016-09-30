local ffi = require 'ffi'
local RReLU, parent = torch.class('nn.RReLU', 'nn.Module')

function RReLU:__init(l, u, ip, cw)
   parent.__init(self)
   self.lower = l or 1/8
   self.upper = u or 1/3
   assert(self.lower <= self.upper and self.lower >= 0 and self.upper >= 0)
   self.noise = torch.Tensor()
   self.train = true
   self.inplace = ip or false
   self.channelwise = cw or false
end

function RReLU:updateOutput(input)
   local gen = ffi.typeof('THGenerator**')(torch._gen)[0]
   input.THNN.RReLU_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.noise:cdata(),
      self.lower,
      self.upper,
      self.train,
      self.inplace,
      self.channelwise,
      gen
   )
   return self.output
end

function RReLU:updateGradInput(input, gradOutput)
   input.THNN.RReLU_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.noise:cdata(),
      self.lower,
      self.upper,
      self.train,
      self.inplace,
      self.channelwise
   )
   return self.gradInput
end

function RReLU:__tostring__()
  return string.format('%s (l:%f, u:%f, channel-wise:%s)', torch.type(self), self.lower, self.upper, self.channelwise)
end

function RReLU:clearState()
   if self.noise then self.noise:set() end
   return parent.clearState(self)
end
