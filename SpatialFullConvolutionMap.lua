local SpatialFullConvolutionMap, parent = torch.class('nn.SpatialFullConvolutionMap', 'nn.Module')

function SpatialFullConvolutionMap:__init(conMatrix, kW, kH, dW, dH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH
   self.connTable = conMatrix
   self.nInputPlane = self.connTable:select(2,1):max()
   self.nOutputPlane = self.connTable:select(2,2):max()

   self.weight = torch.Tensor(self.connTable:size(1), kH, kW)
   self.gradWeight = torch.Tensor(self.connTable:size(1), kH, kW)

   self.bias = torch.Tensor(self.nOutputPlane)
   self.gradBias = torch.Tensor(self.nOutputPlane)
   
   self:reset()
end

function SpatialFullConvolutionMap:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
      self.weight:apply(function()
			   return torch.uniform(-stdv, stdv)
			end)
      self.bias:apply(function()
			 return torch.uniform(-stdv, stdv)
		      end)
   else
      local ninp = torch.Tensor(self.nOutputPlane):zero()
      for i=1,self.connTable:size(1) do ninp[self.connTable[i][2]] =  ninp[self.connTable[i][2]]+1 end
      for k=1,self.connTable:size(1) do
         stdv = 1/math.sqrt(self.kW*self.kH*ninp[self.connTable[k][2]])
         self.weight:select(1,k):apply(function() return torch.uniform(-stdv,stdv) end)
      end
      for k=1,self.bias:size(1) do
         stdv = 1/math.sqrt(self.kW*self.kH*ninp[k])
         self.bias[k] = torch.uniform(-stdv,stdv)
      end

   end
end

function SpatialFullConvolutionMap:updateOutput(input)
   input.nn.SpatialFullConvolutionMap_updateOutput(self, input)
   return self.output
end

function SpatialFullConvolutionMap:updateGradInput(input, gradOutput)
   input.nn.SpatialFullConvolutionMap_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

function SpatialFullConvolutionMap:accGradParameters(input, gradOutput, scale)
   return input.nn.SpatialFullConvolutionMap_accGradParameters(self, input, gradOutput, scale)
end
