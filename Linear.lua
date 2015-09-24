local Linear, parent = torch.class('nn.Linear', 'nn.Module')

function Linear:__init(inputSize, outputSize)
   parent.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize)

   self:reset()
end

function Linear:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
         self.bias[i] = torch.uniform(-stdv, stdv)
      end
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end

   return self
end

function Linear:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      self.output:copy(self.bias)
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.bias:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      if not self.addBuffer or self.addBuffer:nElement() ~= nframe then
         self.addBuffer = input.new(nframe):fill(1)
      end
      self.output:addmm(0, self.output, 1, input, self.weight:t())
      self.output:addr(1, self.addBuffer, self.bias)
   elseif input:dim() == 3 then
      local nframe = input:size(1)
      local nLength = input:size(2)
      local nDimOut = self.weight:size(1)
      local nDimIn = self.weight:size(2)
      local nElement = self.output:nElement()
      self.output:resize(nframe * nLength, nDimOut)
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      if not self.addBuffer or self.addBuffer:nElement() ~= (nframe * nLength) then
         self.addBuffer = input.new(nframe * nLength):fill(1)
      end
      self.output:addmm(0, self.output, 1, input:resize(nframe * nLength, nDimIn), self.weight:t())
      self.output:addr(1, self.addBuffer, self.bias)
      self.output:resize(nframe, nLength, nDimOut)
      input:resize(nframe, nLength, nDimIn)
   else
      error('input must be vector or matrix or 3D tensor')
   end

   return self.output
end

function Linear:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.weight)
      elseif input:dim() == 3 then
         local nframe = gradOutput:size(1)
         local nLength = gradOutput:size(2)
         local nDimOut = self.weight:size(1)
         local nDimIn = self.weight:size(2)
         self.gradInput:resize(nframe * nLength, nDimIn)
         self.gradInput:addmm(0, 1, gradOutput:resize(nframe * nLength, nDimOut), self.weight)
         self.gradInput:resize(nframe, nLength, nDimIn)
         gradOutput:resize(nframe, nLength, nDimOut)
      end

      return self.gradInput
   end
end

function Linear:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      self.gradBias:add(scale, gradOutput)
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
      self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
   elseif input:dim() == 3 then
      local nframe = gradOutput:size(1)
      local nLength = gradOutput:size(2)
      local nDimOut = self.weight:size(1)
      local nDimIn = self.weight:size(2)
      gradOutput:resize(nframe * nLength, nDimOut)
      input:resize(nframe * nLength, nDimIn)
      self.gradWeight:addmm(scale, gradOutput:t(), input)
      self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
      gradOutput:resize(nframe, nLength, nDimOut)
      input:resize(nframe, nLength, nDimIn)
   end
end

-- we do not need to accumulate parameters when sharing
Linear.sharedAccUpdateGradParameters = Linear.accUpdateGradParameters


function Linear:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
