local HardTanh = torch.class('nn.HardTanh', 'nn.Module')

function HardTanh:updateOutput(input)
   return input.nn.HardTanh_updateOutput(self, input)
end

function HardTanh:updateGradInput(input, gradOutput)
   return input.nn.HardTanh_updateGradInput(self, input, gradOutput)
end
