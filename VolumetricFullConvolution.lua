local VolumetricFullConvolution, parent = torch.class('nn.VolumetricFullConvolution', 'nn.Module')

function VolumetricFullConvolution:__init(nInputPlane, nOutputPlane, kT, kH, kW, dT, dH, dW, pT, pH, pW)
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

function VolumetricFullConvolution:reset(stdv)
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

function VolumetricFullConvolution:updateOutput(input)
   input.THNN.VolumetricFullConvolution_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.weight:cdata(),
      self.bias:cdata(),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.dT, self.dW, self.dH,
      self.pT, self.pW, self.pH
   )
   return self.output
end

function VolumetricFullConvolution:updateGradInput(input, gradOutput)
   input.THNN.VolumetricFullConvolution_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.weight:cdata(),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.dT, self.dW, self.dH,
      self.pT, self.pW, self.pH
   )
   return self.gradInput
end

function VolumetricFullConvolution:accGradParameters(input, gradOutput, scale)
   input.THNN.VolumetricFullConvolution_accGradParameters(
      input:cdata(),
      gradOutput:cdata(),
      self.gradWeight:cdata(),
      self.gradBias:cdata(),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.dT, self.dW, self.dH,
      self.pT, self.pW, self.pH,
      scale or 1
   )
end
