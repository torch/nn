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
      local nunit = self.bias:size(1)
      self.output:resize(nframe, nunit)
      if not self.addBuffer or self.addBuffer:nElement() ~= nframe then
         self.addBuffer = input.new(nframe):fill(1)
      end
      if nunit == 1 then
         -- Special case to fix output size of 1 bug:
         self.output:copy(self.bias:view(1,nunit):expand(#self.output))
         self.output:select(2,1):addmv(1, input, self.weight:select(1,1))
      else
         self.output:zero():addr(1, self.addBuffer, self.bias)
         self.output:addmm(1, input, self.weight:t())
      end
   else
      error('input must be vector or matrix')
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
      local nunit = self.bias:size(1)

      if nunit == 1 then
         -- Special case to fix output size of 1 bug:
         self.gradWeight:select(1,1):addmv(scale, input:t(), gradOutput:select(2,1))
         self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
      else
         self.gradWeight:addmm(scale, gradOutput:t(), input)
         self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
      end
   end

end

-- we do not need to accumulate parameters when sharing
Linear.sharedAccUpdateGradParameters = Linear.accUpdateGradParameters


function Linear:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
