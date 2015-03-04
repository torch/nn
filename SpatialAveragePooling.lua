local SpatialAveragePooling, parent = torch.class('nn.SpatialAveragePooling', 'nn.Module')

function SpatialAveragePooling:__init(kW, kH, dW, dH)
   parent.__init(self)

   self.kW = kW
   self.kH = kH
   self.dW = dW or 1
   self.dH = dH or 1
   self.divide = true
end

function SpatialAveragePooling:updateOutput(input)
   input.nn.SpatialAveragePooling_updateOutput(self, input)
   -- for backward compatibility with saved models
   -- which are not supposed to have "divide" field
   if not self.divide then
     self.output:mul(self.kW*self.kH)
   end
   return self.output
end

function SpatialAveragePooling:updateGradInput(input, gradOutput)
   if self.gradInput then
      input.nn.SpatialAveragePooling_updateGradInput(self, input, gradOutput)
      -- for backward compatibility
      if not self.divide then
	self.gradInput:mul(self.kW*self.kH)
      end
      return self.gradInput
   end
end
