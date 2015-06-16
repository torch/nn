local LookupTable, parent = torch.class('nn.LookupTable', 'nn.Module')

LookupTable.__version = 3

function LookupTable:__init(nIndex, nOutput)
   parent.__init(self)

   self.weight = torch.Tensor(nIndex, nOutput)
   self.gradWeight = torch.Tensor(nIndex, nOutput):zero()
   self._count = torch.IntTensor()
   self._input = torch.LongTensor()

   self.shouldScaleGradByFreq = false

   self:reset()
end

function LookupTable:accUpdateOnly()
   self.gradWeight = nil
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
      self._input:resize(input:size()):copy(input)
      return self._input
   end
   return input
end

function LookupTable:updateOutput(input)
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
   input = self:makeInputContiguous(input)
   self.gradWeight.nn.LookupTable_accGradParameters(self, input, gradOutput, scale)
end

function LookupTable:type(type)
   parent.type(self, type)

   if type == 'torch.CudaTensor' then
      -- CUDA uses _sorted and _indices temporary tensors
      self._sorted = self.weight.new()
      self._indices = self.weight.new()
   else
      -- self._count and self._input should only be converted if using Cuda
      self._count = torch.IntTensor()
      self._input = torch.LongTensor()
   end

   return self
end

-- we do not need to accumulate parameters when sharing
LookupTable.sharedAccUpdateGradParameters = LookupTable.accUpdateGradParameters
