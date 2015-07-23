local Min, parent = torch.class('nn.Min', 'nn.Module')

function Min:__init(dimension)
   parent.__init(self)
   dimension = dimension or 1
   self.dimension = dimension
   self._indices = torch.LongTensor()
   self._output = torch.Tensor()
end

function Min:updateOutput(input)
   torch.min(self._output, self._indices, input, self.dimension)
   if input:dim() > 1 then
     self.output = self._output:select(self.dimension, 1)
   else
     self.output = self._output
   end
   return self.output
end

function Min:updateGradInput(input, gradOutput)
   local gradOutputView
   gradOutputView = gradOutput:viewAs(self._output)
   self.gradInput:resizeAs(input):zero():scatter(self.dimension, self._indices, gradOutputView)
   return self.gradInput
end

function Min:type(type)
  -- torch.min expects a LongTensor as indices, whereas cutorch.max expects a CudaTensor.
  if type == 'torch.CudaTensor' then
    parent.type(self, type)
  else
    -- self._indices must be a LongTensor. Setting it to nil temporarily avoids
    -- unnecessary memory allocations.
    local indices
    indices, self._indices = self._indices, nil
    parent.type(self, type)
    self._indices = indices:long()
  end
  return self
end
