local Sigmoid = torch.class('nn.Sigmoid', 'nn.Module')

function Sigmoid:updateOutput(input)
   return input.nn.Sigmoid_updateOutput(self, input)
end

function Sigmoid:updateGradInput(input, gradOutput)
   return input.nn.Sigmoid_updateGradInput(self, input, gradOutput)
end
