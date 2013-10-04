local LogSoftMax = torch.class('nn.LogSoftMax', 'nn.Module')

function LogSoftMax:updateOutput(input)
   return input.nn.LogSoftMax_updateOutput(self, input)
end

function LogSoftMax:updateGradInput(input, gradOutput)
   return input.nn.LogSoftMax_updateGradInput(self, input, gradOutput)
end
