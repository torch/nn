local Select, parent = torch.class('nn.Select', 'nn.Module')

function Select:__init(dimension,index)
   parent.__init(self)
   self.dimension = dimension
   self.index = index 
end

function Select:updateOutput(input)
   local index = self.index < 0 and input:size(self.dimension) + self.index + 1 or self.index
   local output = input:select(self.dimension, index);
   self.output:resizeAs(output)
   return self.output:copy(output)
end

function Select:updateGradInput(input, gradOutput)
   local index = self.index < 0 and input:size(self.dimension) + self.index + 1 or self.index
   self.gradInput:resizeAs(input)  
   self.gradInput:zero()
   self.gradInput:select(self.dimension,index):copy(gradOutput) 
   return self.gradInput
end 
