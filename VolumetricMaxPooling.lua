local VolumetricMaxPooling, parent = torch.class('nn.VolumetricMaxPooling', 'nn.Module')

function VolumetricMaxPooling:__init(kT, kW, kH, dT, dW, dH)
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

   self.indices = torch.Tensor()
end

function VolumetricMaxPooling:updateOutput(input)
   input.nn.VolumetricMaxPooling_updateOutput(self, input)
   return self.output
end

function VolumetricMaxPooling:updateGradInput(input, gradOutput)
   input.nn.VolumetricMaxPooling_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

function VolumetricMaxPooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.indices:resize()
   self.indices:storage():resize(0)
end
