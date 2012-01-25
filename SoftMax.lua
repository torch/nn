local SoftMax, parent = torch.class('nn.SoftMax', 'nn.Module')

function SoftMax:updateOutput(input)
   return input.nn.SoftMax_updateOutput(self, input)
end

function SoftMax:updateGradInput(input, gradOutput)
   return input.nn.SoftMax_updateGradInput(self, input, gradOutput)
end
