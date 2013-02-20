local SparseLinear, parent = torch.class('nn.SparseLinear', 'nn.Module')

function SparseLinear:__init(inputSize, outputSize)
   parent.__init(self)

   self.weightDecay = 0
   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize)
   self.lastInput = torch.Tensor()
   -- state
   self.gradInput:resize(inputSize)
   self.output:resize(outputSize)

   self:reset()
end

function SparseLinear:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(1))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
         self.bias[i] = torch.uniform(-stdv, stdv) * 0.000001
      end
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv):mul(0.000001)
   end
end

function SparseLinear:updateOutput(input)
   return input.nn.SparseLinear_updateOutput(self, input)
end

function SparseLinear:accGradParameters(input, gradOutput, scale)
   return input.nn.SparseLinear_accGradParameters(self, input, gradOutput, scale)
end
