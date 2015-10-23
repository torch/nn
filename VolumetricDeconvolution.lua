local VolumetricDeconvolution, parent = torch.class('nn.VolumetricDeconvolution', 'nn.Module')

function VolumetricDeconvolution:__init(nInputPlane, nOutputPlane, kT, kH, kW, dT, dH, dW, pT, pH, pW)
   parent.__init(self)

   dT = dT or 1
   dW = dW or 1
   dH = dH or 1

   pT = pT or 0
   pW = pW or 0
   pH = pH or 0

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kT = kT
   self.kW = kW
   self.kH = kH
   self.dT = dT
   self.dW = dW
   self.dH = dH
   self.pT = pT
   self.pW = pW
   self.pH = pH

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kT, kH, kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kT, kH, kW)
   self.gradBias = torch.Tensor(nOutputPlane)
   -- temporary buffers for unfolding (CUDA)
   self.finput = torch.Tensor()
   self.fgradInput = torch.Tensor()
   self:reset()
end

function VolumetricDeconvolution:reset(stdv)
  -- initialization of parameters
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

function VolumetricDeconvolution:updateOutput(input)
   return input.nn.VolumetricDeconvolution_updateOutput(self, input)
end

function VolumetricDeconvolution:updateGradInput(input, gradOutput)
   return input.nn.VolumetricDeconvolution_updateGradInput(self, input, gradOutput)
end

function VolumetricDeconvolution:accGradParameters(input, gradOutput, scale)
   return input.nn.VolumetricDeconvolution_accGradParameters(self, input, gradOutput, scale)
end
