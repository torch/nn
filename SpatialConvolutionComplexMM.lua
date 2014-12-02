local SpatialConvolutionComplexMM, parent = torch.class('nn.SpatialConvolutionComplexMM', 'nn.Module')

function SpatialConvolutionComplexMM:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padding)
   parent.__init(self)
   
   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH

   self.dW = dW
   self.dH = dH
   self.padding = padding or 0

   self.weight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW*2)
   self.bias = torch.Tensor(nOutputPlane*2)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW*2)
   self.gradBias = torch.Tensor(nOutputPlane*2)

   self.finput = torch.Tensor()
   self.fgradInput = torch.Tensor()
   
   self:reset()
end

function SpatialConvolutionComplexMM:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane*2)
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

function SpatialConvolutionComplexMM:updateOutput(input)
   return input.nn.SpatialConvolutionComplexMM_updateOutput(self, input)
end

function SpatialConvolutionComplexMM:updateGradInput(input, gradOutput)
   if self.gradInput then
      return input.nn.SpatialConvolutionComplexMM_updateGradInput(self, input, gradOutput)
   end
end

function SpatialConvolutionComplexMM:accGradParameters(input, gradOutput, scale)
   return input.nn.SpatialConvolutionComplexMM_accGradParameters(self, input, gradOutput, scale)
end
