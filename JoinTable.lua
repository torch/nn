local JoinTable, parent = torch.class('nn.JoinTable', 'nn.Module')

function JoinTable:__init(dimension)
   parent.__init(self)
   self.size = torch.LongStorage()
   self.dimension = dimension
   self.gradInput = {}
end 

function JoinTable:updateOutput(input) 
   for i=1,#input do
      local currentOutput = input[i]
      if i == 1 then
         self.size:resize(currentOutput:dim()):copy(currentOutput:size())
      else
         self.size[self.dimension] = self.size[self.dimension] 
            + currentOutput:size(self.dimension)
      end 
   end
   self.output:resize(self.size)
   
   local offset = 1  
   for i=1,#input do
      local currentOutput = input[i]
      self.output:narrow(self.dimension, offset, 
			 currentOutput:size(self.dimension)):copy(currentOutput)
      offset = offset + currentOutput:size(self.dimension)
   end
   return self.output

end

function JoinTable:updateGradInput(input, gradOutput)
   for i=1,#input do 
      if self.gradInput[i] == nil then
         self.gradInput[i] = input[i].new()
      end
      self.gradInput[i]:resizeAs(input[i])
   end

   local offset = 1
   for i=1,#input do
      local currentOutput = input[i] 
      local currentGradInput = gradOutput:narrow(self.dimension, offset, 
					  currentOutput:size(self.dimension))
      self.gradInput[i]:copy(currentGradInput)
      offset = offset + currentOutput:size(self.dimension)
   end
   return self.gradInput
end
