local Sum, parent = torch.class('nn.Sum', 'nn.Module')

function Sum:__init(dimension, nInputDims)
   parent.__init(self)
   dimension = dimension or 1
   self.dimension = dimension
   -- do not assign default value to nInputDims or it will break backward compatibility
   self.nInputDims = nInputDims
end

function Sum:_getPositiveDimension(input)
   local dimension = self.dimension
   if dimension < 0 then
      dimension = input:dim() + dimension + 1
   elseif self.nInputDims and input:dim()==(self.nInputDims+1) then
      dimension = dimension + 1
   end
   return dimension
end

function Sum:updateOutput(input)
   local dimension = self:_getPositiveDimension(input)
   if type(self.output) == 'number' then
      self.output = input.new()
   end
   self.output:sum(input, dimension)
   if self.output:nDimension() > 1 then
      self.output = self.output:select(dimension, 1)
   end
   return self.output
end

function Sum:updateGradInput(input, gradOutput)
   local dimension = self:_getPositiveDimension(input)
   -- zero-strides dont work with MKL/BLAS, so
   -- dont set self.gradInput to zero-stride tensor.
   -- Instead, do a deepcopy
   local size = input:size()
   size[dimension] = 1
   gradOutput = gradOutput:view(size)
   self.gradInput:resizeAs(input)
   self.gradInput:copy(gradOutput:expandAs(input))

   return self.gradInput
end
