local Linear, parent = torch.class('nn.Linear', 'nn.Module')

function Linear:__init(inputSize, outputSize)
   parent.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize)
   
   self:setup()
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
end

function Linear:setup()
   -- for backwards compatibility
   if not self.v2 then
      self._ones = self.weight.new{1}
      self.v2 = true
   end
end

function Linear:updateOutput(input)
   self:setup()
   return input.nn.Linear_updateOutput(self, input)
end

function Linear:updateGradInput(input, gradOutput)
   if self.gradInput then
      self:setup()
      return input.nn.Linear_updateGradInput(self, input, gradOutput)
   end
end

function Linear:accGradParameters(input, gradOutput, scale)
   self:setup()
   return input.nn.Linear_accGradParameters(self, input, gradOutput, scale)
end

-- we do not need to accumulate parameters when sharing
Linear.sharedAccUpdateGradParameters = Linear.accUpdateGradParameters
