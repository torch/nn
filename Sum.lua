local Sum, parent = torch.class('nn.Sum', 'nn.Module')

function Sum:__init(dimension)
   parent.__init(self)
   dimension = dimension or 1
   self.dimension = dimension
end

function Sum:updateOutput(input)
   if type(self.output) == 'number' then
      self.output = input.new()
   end
   self.output:sum(input, self.dimension)
   if self.output:nDimension() > 1 then
      self.output = self.output:select(self.dimension, 1)
   end
   return self.output
end

function Sum:updateGradInput(input, gradOutput)
    -- zero-strides dont work with MKL/BLAS, so
    -- dont set self.gradInput to zero-stride tensor.
    -- Instead, do a deepcopy
    local size = input:size()
    size[self.dimension] = 1
    gradOutput = gradOutput:view(size)
    self.gradInput:resizeAs(input)
    self.gradInput:copy(gradOutput:expandAs(input))

    return self.gradInput
end
