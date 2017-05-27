--[[ An nn.Module that splits a tensor into evenly sized sub-tensors
numerically indexed in a lua table.  The sub-tensors will be split along
the dimension corresponding to the first argument of the constructor.

This model is the inverse of JoinTable.

Example:

> require 'nn'
> require 'torch'
> st = nn.SplitTable(1, 2)
> t = torch.rand(2,2)
> =t
 0.7110  0.1456
 0.2049  0.1312
  [torch.DoubleTensor of size 2x2]

> require 'pprint'
> pprint(st:updateOutput(t))
{ 0.7110
  0.1456
 [torch.DoubleTensor of size 2],
  0.2049
  0.1312
  [torch.DoubleTensor of size 2]
}

]]

local SplitTable, parent = torch.class('nn.SplitTable', 'nn.Module')

function SplitTable:__init(dimension, nInputDims)
   parent.__init(self)
   self.dimension = dimension
   self.nInputDims = nInputDims
end

function SplitTable:_getPositiveDimension(input)
   local dimension = self.dimension
   if dimension < 0 then
      dimension = input:dim() + dimension + 1
   elseif self.nInputDims and input:dim()==(self.nInputDims+1) then
      dimension = dimension + 1
   end
   return dimension
end

function SplitTable:updateOutput(input)
   local dimension = self:_getPositiveDimension(input)
   local slices = input:size(dimension)

   local currentOutput= {}
   for i=1,slices do
      currentOutput[#currentOutput+1] = input:select(dimension,i)
   end
   self.output = currentOutput
   return self.output
end 

function SplitTable:updateGradInput(input, gradOutput)
   local dimension = self:_getPositiveDimension(input)
   local slices = input:size(dimension)
   if self.gradInput then
      self.gradInput:resizeAs(input)

      for i=1,slices do 
         local currentGradInput = gradOutput[i];        
         self.gradInput:select(dimension,i):copy(currentGradInput)
      end
   end
   return self.gradInput
end
