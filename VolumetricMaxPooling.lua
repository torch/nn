local VolumetricMaxPooling, parent = torch.class('nn.VolumetricMaxPooling', 'nn.Module')

function VolumetricMaxPooling:__init(kT, kW, kH, dT, dW, dH, padT, padW, padH)
VolumetricMaxPooling.__version = 2

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

   self.padT = padT or 0
   self.padW = padW or 0
   self.padH = padH or 0


   self.ceil_mode = false
   self.indices = torch.Tensor()
end

function VolumetricMaxPooling:ceil()
    self.ceil_mode = true
    return self
end

function VolumetricMaxPooling:floor()
    self.ceil_mode = false
    return self
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

function VolumetricMaxPooling:read(file, version)
   parent.read(self, file)
   if version < 2 then
      self.ceil_mode = false
   end
end
