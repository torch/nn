local VolumetricConvolution, parent = torch.class('nn.VolumetricConvolution', 'nn.Module')

function VolumetricConvolution:__init(nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH)
   parent.__init(self)

   dT = dT or 1
   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kT = kT
   self.kW = kW
   self.kH = kH
   self.dT = dT
   self.dW = dW
   self.dH = dH

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kT, kH, kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kT, kH, kW)
   self.gradBias = torch.Tensor(nOutputPlane)
   
   self:reset()
end

function VolumetricConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kT*self.kW*self.kH*self.nInputPlane)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
end

function VolumetricConvolution:updateOutput(input)
   return input.nn.VolumetricConvolution_updateOutput(self, input)
end

function VolumetricConvolution:updateGradInput(input, gradOutput)
   return input.nn.VolumetricConvolution_updateGradInput(self, input, gradOutput)
end

function VolumetricConvolution:accGradParameters(input, gradOutput, scale)
   return input.nn.VolumetricConvolution_accGradParameters(self, input, gradOutput, scale)
end
