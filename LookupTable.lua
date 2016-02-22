local THNN = require 'nn.THNN'
local LookupTable, parent = torch.class('nn.LookupTable', 'nn.Module')

LookupTable.__version = 4

function LookupTable:__init(nIndex, nOutput, paddingValue)
   parent.__init(self)

   self.weight = torch.Tensor(nIndex, nOutput)
   self.gradWeight = torch.Tensor(nIndex, nOutput):zero()
   self.paddingValue = paddingValue or 0

   self:reset()
end

function LookupTable:backCompatibility()
   self._count = self._count or torch.IntTensor()
   self._input = self._input or torch.LongTensor()

   if not self.shouldScaleGradByFreq then
      self.shouldScaleGradByFreq = false
   end
end

function LookupTable:accUpdateOnly()
   self.gradWeight = nil
   return self
end

function LookupTable:setPadding(paddingValue)
    self.paddingValue = paddingValue
    return self
end

function LookupTable:scaleGradByFreq()
   self.shouldScaleGradByFreq = true
   return self
end

function LookupTable:reset(stdv)
   stdv = stdv or 1
   self.weight:normal(0, stdv)
end

function LookupTable:makeInputContiguous(input)
   -- make sure input is a contiguous torch.LongTensor
   if (not input:isContiguous()) or torch.type(input) ~= torch.type(self._input) then
      self.copiedInput = true
      self._input:resize(input:size()):copy(input)
      return self._input
   end
   self.copiedInput = false
   return input
end

function LookupTable:updateOutput(input)
   self:backCompatibility()
   input = self:makeInputContiguous(input)
   if input:dim() == 1 then
      self.output:index(self.weight, 1, input)
   elseif input:dim() == 2 then
      self.output:index(self.weight, 1, input:view(-1))
      self.output = self.output:view(input:size(1), input:size(2), self.weight:size(2))
   else
      error("input must be a vector or matrix")
   end
   return self.output
end

function LookupTable:accGradParameters(input, gradOutput, scale)
   self:backCompatibility()
   input = self.copiedInput and self._input or input
   if input:dim() == 2 then
      input = input:view(-1)
   elseif input:dim() ~= 1 then
      error("input must be a vector or matrix")
   end

   if not gradOutput:isContiguous() then
       self._gradOutput = self._gradOutput or gradOutput.new()
       self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
       gradOutput = self._gradOutput
   end

   self.gradWeight.THNN.LookupTable_accGradParameters(
      input:cdata(),
      gradOutput:cdata(),
      self.gradWeight:cdata(),
      self._count:cdata(),
      THNN.optionalTensor(self._sorted),
      THNN.optionalTensor(self._indices),
      self.shouldScaleGradByFreq or false,
      self.paddingValue or 0,
      scale or 1
   )
end

function LookupTable:type(type, tensorCache)
   parent.type(self, type, tensorCache)

   if type == 'torch.CudaTensor' then
      -- CUDA uses _sorted and _indices temporary tensors
      self._sorted = self.weight.new()
      self._indices = self.weight.new()
      self._count = self.weight.new()
      self._input = self.weight.new()
   else
      -- self._count and self._input should only be converted if using Cuda
      self._count = torch.IntTensor()
      self._input = torch.LongTensor()
   end

   return self
end

function LookupTable:clearState()
   self._gradOutput = nil
   return self
end

-- we do not need to accumulate parameters when sharing
LookupTable.sharedAccUpdateGradParameters = LookupTable.accUpdateGradParameters
