local Mean, parent = torch.class('nn.Mean', 'nn.Module')

function Mean:__init(dimension)
   parent.__init(self)
   dimension = dimension or 1
   self.dimension = dimension
   self._gradInput = torch.Tensor()
end

function Mean:updateOutput(input)
   self.output:mean(input, self.dimension)
   if self.output:nDimension() > 1 then
      self.output = self.output:select(self.dimension, 1)
   end
   return self.output
end

function Mean:updateGradInput(input, gradOutput)
   self._gradInput:resizeAs(gradOutput):copy(gradOutput)
   self._gradInput:mul(1/input:size(self.dimension))

   if input:nDimension() > 1 then
      self._gradInput = nn.utils.addSingletonDimension(self._gradInput,
                                                       self.dimension)
   end
   self.gradInput = self._gradInput:expandAs(input)
   return self.gradInput
end
