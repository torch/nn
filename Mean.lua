local Mean, parent = torch.class('nn.Mean', 'nn.Module')

function Mean:__init(dimension)
   parent.__init(self)
   dimension = dimension or 1
   self.dimension = dimension
end

function Mean:updateOutput(input)
   self.output:mean(input, self.dimension)
   if self.output:nDimension() > 1 then
      self.output = self.output:select(self.dimension, 1)
   end
   return self.output
end

function Mean:updateGradInput(input, gradOutput)
   local size = gradOutput:size():totable()
   local stride = gradOutput:stride():totable()

   if input:nDimension() > 1 then
      table.insert(size, self.dimension, input:size(self.dimension))
      table.insert(stride, self.dimension, 0)
   else
      size[1] = input:size(1)
      stride[1] = 0
   end

   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   self.gradInput:mul(1/input:size(self.dimension))
   self.gradInput:resize(torch.LongStorage(size), torch.LongStorage(stride))

   return self.gradInput
end
