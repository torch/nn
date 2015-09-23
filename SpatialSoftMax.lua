local SpatialSoftMax, _ = torch.class('nn.SpatialSoftMax', 'nn.Module')

function SpatialSoftMax:updateOutput(input)
   return input.nn.SoftMax_updateOutput(self, input)
end

function SpatialSoftMax:updateGradInput(input, gradOutput)
   return input.nn.SoftMax_updateGradInput(self, input, gradOutput)
end
