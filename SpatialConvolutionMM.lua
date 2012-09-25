local SpatialConvolutionMM, parent = torch.class('nn.SpatialConvolutionMM', 'nn.Module')

function SpatialConvolutionMM:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH

   self.weight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
   self.gradBias = torch.Tensor(nOutputPlane)

   self.finput = torch.Tensor()
   self.fgradInput = torch.Tensor()
   
   self:reset()
end

function SpatialConvolutionMM:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   self.weight:apply(function()
                        return torch.uniform(-stdv, stdv)
                     end)
   self.bias:apply(function()
                      return torch.uniform(-stdv, stdv)
                   end)   
end

function SpatialConvolutionMM:updateOutput(input)
   return input.nn.SpatialConvolutionMM_updateOutput(self, input)
end

function SpatialConvolutionMM:updateGradInput(input, gradOutput)
   if self.gradInput then
      return input.nn.SpatialConvolutionMM_updateGradInput(self, input, gradOutput)
   end
end

function SpatialConvolutionMM:accGradParameters(input, gradOutput, scale)
   return input.nn.SpatialConvolutionMM_accGradParameters(self, input, gradOutput, scale)
end
