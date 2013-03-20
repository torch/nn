local SpatialConvolutionCUDA, parent = torch.class('nn.SpatialConvolutionCUDA', 'nn.Module')

function SpatialConvolutionCUDA:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH

   self.weight = torch.Tensor(nInputPlane, kH, kW, nOutputPlane)
   self.gradWeight = torch.Tensor(nInputPlane, kH, kW, nOutputPlane)
   
   self:reset()
end

function SpatialConvolutionCUDA:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   self.weight:uniform(-stdv, stdv)
end

function SpatialConvolutionCUDA:updateOutput(input)
   return input.nn.SpatialConvolutionCUDA_updateOutput(self, input)
end

function SpatialConvolutionCUDA:updateGradInput(input, gradOutput)
   return input.nn.SpatialConvolutionCUDA_updateGradInput(self, input, gradOutput)
end

function SpatialConvolutionCUDA:accGradParameters(input, gradOutput, scale)
   return input.nn.SpatialConvolutionCUDA_accGradParameters(self, input, gradOutput, scale)
end

function SpatialConvolutionCUDA:copy(sc)
   local weight = sc.weight:clone()
   weight:resize(sc.nOutputPlane, sc.nInputPlane * sc.kH * sc.kW)
   weight = weight:t():contiguous()
   weight:resize(sc.nInputPlane, sc.kH, sc.kW, sc.nOutputPlane)
   self.weight:copy(weight)
end

