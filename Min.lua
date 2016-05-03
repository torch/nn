local Min, parent = torch.class('nn.Min', 'nn.Module')

function Min:__init(dimension, nInputDims)
   parent.__init(self)
   dimension = dimension or 1
   self.dimension = dimension
   -- do not assign default value to nInputDims or it will break backward compatibility
   self.nInputDims = nInputDims
end

function Min:_getPositiveDimension(input)
   local dimension = self.dimension
   if dimension < 0 then
      dimension = input:dim() + dimension + 1
   elseif self.nInputDims and input:dim()==(self.nInputDims+1) then
      dimension = dimension + 1
   end
   return dimension
end

function Min:_lazyInit()
   self._output = self._output or self.output.new()
   self._indices = self._indices or
      (torch.type(self.output) == 'torch.CudaTensor' and torch.CudaTensor() or torch.LongTensor())
end

function Min:updateOutput(input)
   self:_lazyInit()
   local dimension = self:_getPositiveDimension(input)
   torch.min(self._output, self._indices, input, dimension)
   if input:dim() > 1 then
     self.output:set(self._output:select(dimension, 1))
   else
     self.output:set(self._output)
   end
   return self.output
end

function Min:updateGradInput(input, gradOutput)
   self:_lazyInit()
   local dimension = self:_getPositiveDimension(input)
   local gradOutputView
   if input:dim() > 1 then
     gradOutputView = nn.utils.addSingletonDimension(gradOutput, dimension)
   else
     gradOutputView = gradOutput
   end
   self.gradInput:resizeAs(input):zero():scatter(dimension, self._indices, gradOutputView)
   return self.gradInput
end

function Min:type(type, tensorCache)
  -- torch.min expects a LongTensor as indices, whereas cutorch.max expects a CudaTensor.
  if type == 'torch.CudaTensor' then
    parent.type(self, type, tensorCache)
  else
    -- self._indices must be a LongTensor. Setting it to nil temporarily avoids
    -- unnecessary memory allocations.
    local indices
    indices, self._indices = self._indices, nil
    parent.type(self, type, tensorCache)
    self._indices = indices and indices:long() or nil
  end
  return self
end

function Min:clearState()
   nn.utils.clear(self, '_indices', '_output')
   return parent.clearState(self)
end
