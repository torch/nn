local SplitTable, parent = torch.class('nn.SplitTable', 'nn.Module')

function SplitTable:__init(dimension)
   parent.__init(self)
   self.modules = {} 
   self.dimension = dimension
end

function SplitTable:updateOutput(input)
   local currentOutput= {};
   local slices = input:size(self.dimension)
   for i=1,slices do
      currentOutput[#currentOutput+1] = input:select(self.dimension,i)
   end
   self.output = currentOutput
   return self.output
end 


function SplitTable:updateGradInput(input, gradOutput)
   local slices = input:size(self.dimension)
   self.gradInput:resizeAs(input)

   local offset = 1
   for i=1,slices do 
      local currentGradInput = gradOutput[i];        
      self.gradInput:select(self.dimension,i):copy(currentGradInput)
   end
   return self.gradInput
end
