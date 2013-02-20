local SpatialFullConvolution, parent = torch.class('nn.SpatialFullConvolution','nn.Module')

function SpatialFullConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH

   self.weight = torch.Tensor(nInputPlane, nOutputPlane, kH, kW)
   self.gradWeight = torch.Tensor(nInputPlane, nOutputPlane, kH, kW)
   self.bias = torch.Tensor(self.nOutputPlane)
   self.gradBias = torch.Tensor(self.nOutputPlane)

   self:reset()
end

function SpatialFullConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      local nInputPlane = self.nInputPlane
      local kH = self.kH
      local kW = self.kW
      stdv = 1/math.sqrt(kW*kH*nInputPlane)
   end
   self.weight:apply(function()
                        return torch.uniform(-stdv, stdv)
                     end)
   self.bias:apply(function()
                        return torch.uniform(-stdv, stdv)
                     end)
end

function SpatialFullConvolution:updateOutput(input)
   return input.nn.SpatialFullConvolution_updateOutput(self, input)
end

function SpatialFullConvolution:updateGradInput(input, gradOutput)
   if self.gradInput then
      return input.nn.SpatialFullConvolution_updateGradInput(self, input, gradOutput)
   end
end
function SpatialFullConvolution:accGradParameters(input, gradOutput, scale)
   return input.nn.SpatialFullConvolution_accGradParameters(self, input, gradOutput, scale)
end

