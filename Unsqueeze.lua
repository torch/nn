local Unsqueeze, parent = torch.class('nn.Unsqueeze', 'nn.Module')

local function _checkPos(pos)
   local pos = pos or error('the position to insert singleton dim not specified')
   assert(type(pos) == 'number')
   assert(pos > 0)
   return pos
end

local function _assertTensor(t)
   assert(t or torch.isTensor(t), "Unsqueeze only works on tensor")
end

local function _unsqueezeSize(inputSize, numInputDims, pos)
   -- is batchMode?
   local offsetDim = #inputSize - numInputDims
   assert(offsetDim >= 0, "numInputDims must be <= input:dim()")
   -- pos overflow?
   assert(offsetDim + pos <=  1 + #inputSize,
      ("offsetDim + pos (=%d + %d) exceeds input dim (=%d) by more than 1"):format(
         offsetDim, pos, #inputSize)
   )

   local outputSize = {}
   -- left to pos
   for i = 1, offsetDim + pos-1 do
      table.insert(outputSize, inputSize[i])
   end
   -- this pos: the singleton dim
   table.insert(outputSize, 1)
   -- right to pose
   for i = offsetDim + pos, #inputSize do
      table.insert(outputSize, inputSize[i])
   end

   return outputSize
end

function Unsqueeze:__init(pos, numInputDims)
   parent.__init(self)
   self.pos = _checkPos(pos)
   self:setNumInputDims(numInputDims)
end

function Unsqueeze:setNumInputDims(numInputDims)
   self.numInputDims = numInputDims
   return self
end

function Unsqueeze:updateOutput(input)
   _assertTensor(input)

   local inputSize = input:size():totable()
   local numInputDims = self.numInputDims or input:dim()
   local outputSize = _unsqueezeSize(inputSize, numInputDims, self.pos)
   self.output = input:view( table.unpack(outputSize) )
   return self.output
end

function Unsqueeze:updateGradInput(input, gradOutput)
   assert(input:nElement() == gradOutput:nElement())
   self.gradInput = gradOutput:view(input:size())
   return self.gradInput
end

function Unsqueeze:__tostring__()
   return torch.type(self)..'(dim ' .. self.pos .. ')'
end