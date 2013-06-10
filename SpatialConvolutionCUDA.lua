local SpatialConvolutionCUDA, parent = torch.class('nn.SpatialConvolutionCUDA', 'nn.Module')

function SpatialConvolutionCUDA:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padding, partialSum)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1
   partialSum = partialSum or 0

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH
   self.padding = padding or 0
   self.partialSum = partialSum

   self.weight = torch.Tensor(nInputPlane, kH, kW, nOutputPlane)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeightPartial = torch.Tensor(nInputPlane, kH, kW, nOutputPlane)
   self.gradWeight = torch.Tensor(nInputPlane, kH, kW, nOutputPlane)
   self.gradBias = torch.Tensor(nOutputPlane)
  
   self:reset()
end

function SpatialConvolutionCUDA:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv)
end

function SpatialConvolutionCUDA:updateOutput(input)
   input.nn.SpatialConvolutionCUDA_updateOutput(self, input)
   for i = 1,self.nOutputPlane do
      self.output[i]:add(self.bias[i])
   end
   return self.output
end

function SpatialConvolutionCUDA:updateGradInput(input, gradOutput)
   input.nn.SpatialConvolutionCUDA_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

function SpatialConvolutionCUDA:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   input.nn.SpatialConvolutionCUDA_accGradParameters(self, input, gradOutput, scale)
   if self.partialSum > 0 then
      self.gradWeight:add(self.gradWeightPartial:sum(1):resizeAs(self.gradWeight))
   end
   for i = 1,self.nOutputPlane do
      self.gradBias:narrow(1,i,1):add(scale * gradOutput[i]:sum() )
   end
end

-- this routine copies weight+bias from a regular SpatialConvolution module
function SpatialConvolutionCUDA:copy(sc)
   local weight = sc.weight:clone()
   weight:resize(sc.nOutputPlane, sc.nInputPlane * sc.kH * sc.kW)
   weight = weight:t():contiguous()
   weight:resize(sc.nInputPlane, sc.kH, sc.kW, sc.nOutputPlane)
   self.weight:copy(weight)
   self.bias:copy(sc.bias)
end

