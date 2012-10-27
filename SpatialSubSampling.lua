local SpatialSubSampling, parent = torch.class('nn.SpatialSubSampling', 'nn.Module')

function SpatialSubSampling:__init(nInputPlane, kW, kH, dW, dH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH

   self.weight = torch.Tensor(nInputPlane)
   self.bias = torch.Tensor(nInputPlane)
   self.gradWeight = torch.Tensor(nInputPlane)
   self.gradBias = torch.Tensor(nInputPlane)
   
   self:reset()
end

function SpatialSubSampling:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH)
   end
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv)
end

function SpatialSubSampling:updateOutput(input)
   return input.nn.SpatialSubSampling_updateOutput(self, input)
end

function SpatialSubSampling:updateGradInput(input, gradOutput)
   if self.gradInput then
      return input.nn.SpatialSubSampling_updateGradInput(self, input, gradOutput)
   end
end

function SpatialSubSampling:accGradParameters(input, gradOutput, scale)
   return input.nn.SpatialSubSampling_accGradParameters(self, input, gradOutput, scale)
end
