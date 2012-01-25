local Min, parent = torch.class('nn.Min', 'nn.Module')

function Min:__init(dimension)
   parent.__init(self)
   dimension = dimension or 1
   self.dimension = dimension
   self.indices = torch.Tensor()
end

function Min:updateOutput(input)
   return input.nn.Min_updateOutput(self, input)
end

function Min:updateGradInput(input, gradOutput)
   return input.nn.Min_updateGradInput(self, input, gradOutput)
end
