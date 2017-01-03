local LinearTHNN, parent = torch.class('nn.LinearTHNN', 'nn.Module')

function LinearTHNN:__init(inputSize, outputSize, bias)
   parent.__init(self)
   local bias = ((bias == nil) and true) or bias
   self.weight = torch.Tensor(outputSize, inputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   if bias then
      self.bias = torch.Tensor(outputSize)
      self.gradBias = torch.Tensor(outputSize)
   end
   self.addBuffer = torch.Tensor(outputSize)
   self:reset()
end

function LinearTHNN:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function LinearTHNN:reset(stdv)
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
      end
      if self.bias then
         for i=1,self.bias:nElement() do
            self.bias[i] = torch.uniform(-stdv, stdv)
         end
      end
   else
      self.weight:uniform(-stdv, stdv)
      if self.bias then self.bias:uniform(-stdv, stdv) end
   end
   return self
end

function LinearTHNN:updateOutput(input)
   input.THNN.Linear_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.weight:cdata(),
      self.bias and self.bias:cdata(),
      self.addBuffer:cdata()
   )
   return self.output
end

function LinearTHNN:updateGradInput(input, gradOutput)
   input.THNN.Linear_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.weight:cdata()
   )
   return self.gradInput
end

function LinearTHNN:accGradParameters(input, gradOutput, scale)
   input.THNN.Linear_accGradParameters(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.weight:cdata(),
      self.bias and self.bias:cdata(),
      self.gradWeight:cdata(),
      self.bias and self.gradBias:cdata(),
      self.addBuffer:cdata(),
      scale or 1
   )
   return self.gradWeight
end

function LinearTHNN:sharedAccUpdateGradParameters(input, gradOutput, lr)
   -- we do not need to accumulate parameters when sharing:
   self:defaultAccUpdateGradParameters(input, gradOutput, lr)
end

function LinearTHNN:clearState()
   if self.addBuffer then self.addBuffer:set() end
   return parent.clearState(self)
end

function LinearTHNN:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1)) ..
      (self.bias == nil and ' without bias' or '')
end
