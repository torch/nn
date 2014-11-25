local SpatialAveragePooling, parent = torch.class('nn.SpatialAveragePooling', 'nn.Module')

function SpatialAveragePooling:__init(kW, kH, dW, dH)
   parent.__init(self)

   self.kW = kW
   self.kH = kH
   self.dW = dW or 1
   self.dH = dH or 1
end

function SpatialAveragePooling:updateOutput(input)
   return input.nn.SpatialAveragePooling_updateOutput(self, input)
end

function SpatialAveragePooling:updateGradInput(input, gradOutput)
   if self.gradInput then
      return input.nn.SpatialAveragePooling_updateGradInput(self, input, gradOutput)
   end
end
