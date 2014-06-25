local SplitTable, parent = torch.class('nn.SplitTable', 'nn.Module')

function SplitTable:__init(dimension)
   parent.__init(self)
   self.modules = {}
   self.dimension = dimension
   self.nInputDims = nil
end

-- Sets the expected number of dimensions
-- in a non-minibatch input.
function SplitTable:setNumInputDims(nInputDims)
   self.nInputDims = nInputDims
   return self
end

function SplitTable:updateOutput(input)
   local dimension = self.dimension
   if self.nInputDims and input:dim()==(self.nInputDims+1) then
       dimension = dimension + 1
   end
   local currentOutput= {}
   local slices = input:size(dimension)
   for i=1,slices do
      currentOutput[#currentOutput+1] = input:select(dimension,i)
   end
   self.output = currentOutput
   return self.output
end 

function SplitTable:updateGradInput(input, gradOutput)
   local dimension = self.dimension
   if self.nInputDims and input:dim()==(self.nInputDims+1) then
      dimension = dimension + 1
   end
   local slices = input:size(dimension)
   self.gradInput:resizeAs(input)

   local offset = 1
   for i=1,slices do 
      local currentGradInput = gradOutput[i];        
      self.gradInput:select(dimension,i):copy(currentGradInput)
   end
   return self.gradInput
end
