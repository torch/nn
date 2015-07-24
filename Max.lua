local Max, parent = torch.class('nn.Max', 'nn.Module')

function Max:__init(dimension)
   parent.__init(self)
   dimension = dimension or 1
   self.dimension = dimension
end

function Max:_lazyInit()
   self._output = self._output or self.output.new()
   self._indices = self._indices or
      (torch.type(self.output) == 'torch.CudaTensor' and torch.CudaTensor() or torch.LongTensor())
end

function Max:updateOutput(input)
   self:_lazyInit()
   torch.max(self._output, self._indices, input, self.dimension)
   if input:dim() > 1 then
     self.output = self._output:select(self.dimension, 1)
   else
     self.output = self._output
   end
   return self.output
end

function Max:updateGradInput(input, gradOutput)
   self:_lazyInit()
   local gradOutputView
   if input:dim() > 1 then
     gradOutputView = nn.utils.addSingletonDimension(gradOutput, self.dimension)
   else
     gradOutputView = gradOutput
   end
   self.gradInput:resizeAs(input):zero():scatter(self.dimension, self._indices, gradOutputView)
   return self.gradInput
end

function Max:type(type)
  -- torch.max expects a LongTensor as indices, whereas cutorch.max expects a CudaTensor.
  if type == 'torch.CudaTensor' then
    parent.type(self, type)
  else
    -- self._indices must be a LongTensor. Setting it to nil temporarily avoids
    -- unnecessary memory allocations.
    local indices
    indices, self._indices = self._indices, nil
    parent.type(self, type)
    self._indices = indices and indices:long() or nil
  end
  return self
end
