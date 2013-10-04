local Exp = torch.class('nn.Exp', 'nn.Module')

function Exp:updateOutput(input)
   return input.nn.Exp_updateOutput(self, input)
end

function Exp:updateGradInput(input, gradOutput)
   return input.nn.Exp_updateGradInput(self, input, gradOutput)
end
