local SoftPlus = torch.class('nn.SoftPlus', 'nn.Module')

function SoftPlus:updateOutput(input)
   return input.nn.SoftPlus_updateOutput(self, input)
end

function SoftPlus:updateGradInput(input, gradOutput)
   return input.nn.SoftPlus_updateGradInput(self, input, gradOutput)
end
