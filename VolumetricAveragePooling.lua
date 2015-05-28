local VolumetricAveragePooling, parent = torch.class(
   'nn.VolumetricAveragePooling', 'nn.Module')

function VolumetricAveragePooling:__init(kT, kW, kH, dT, dW, dH)
   parent.__init(self)

   dT = dT or kT
   dW = dW or kW
   dH = dH or kH

   self.kT = kT
   self.kH = kH
   self.kW = kW
   self.dT = dT
   self.dW = dW
   self.dH = dH
end

function VolumetricAveragePooling:updateOutput(input)
   input.nn.VolumetricAveragePooling_updateOutput(self, input)
   return self.output
end

function VolumetricAveragePooling:updateGradInput(input, gradOutput)
   input.nn.VolumetricAveragePooling_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

function VolumetricAveragePooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
end
