local Mean, parent = torch.class('nn.Mean', 'nn.Module')

function Mean:__init(dimension, nInputDims)
   parent.__init(self)
   dimension = dimension or 1
   self.dimension = dimension
   -- do not assign default value to nInputDims or it will break backward compatibility
   self.nInputDims = nInputDims
   self._gradInput = torch.Tensor()
end

function Mean:_getPositiveDimension(input)
   local dimension = self.dimension
   if dimension < 0 then
      dimension = input:dim() + dimension + 1
   elseif self.nInputDims and input:dim()==(self.nInputDims+1) then
      dimension = dimension + 1
   end
   return dimension
end

function Mean:updateOutput(input)
   local dimension = self:_getPositiveDimension(input)
   self.output:mean(input, dimension)
   if self.output:nDimension() > 1 then
      self.output = self.output:select(dimension, 1)
   end
   return self.output
end

function Mean:updateGradInput(input, gradOutput)
   local dimension = self:_getPositiveDimension(input)
   self._gradInput:resizeAs(gradOutput):copy(gradOutput)
   self._gradInput:mul(1/input:size(dimension))

   if input:nDimension() > 1 then
      self._gradInput = nn.utils.addSingletonDimension(self._gradInput, dimension)
   end
   self.gradInput = self._gradInput:expandAs(input)
   return self.gradInput
end
