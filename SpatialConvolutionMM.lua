local SpatialConvolutionMM, parent = torch.class('nn.SpatialConvolutionMM', 'nn.Module')

function SpatialConvolutionMM:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   parent.__init(self)
   
   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH

   self.dW = dW
   self.dH = dH
   self.padW = padW or 0
   self.padH = padH or self.padW

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

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput then
      if not gradOutput:isContiguous() then
	 self._gradOutput = self._gradOutput or gradOutput.new()
	 self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
	 gradOutput = self._gradOutput
      end
   end
   return input, gradOutput
end

function SpatialConvolutionMM:updateOutput(input)
   -- backward compatibility
   if self.padding then
      self.padW = self.padding
      self.padH = self.padding
      self.padding = nil
   end
   input = makeContiguous(self, input)
   return input.nn.SpatialConvolutionMM_updateOutput(self, input)
end

function SpatialConvolutionMM:updateGradInput(input, gradOutput)
   if self.gradInput then
      input, gradOutput = makeContiguous(self, input, gradOutput)
      return input.nn.SpatialConvolutionMM_updateGradInput(self, input, gradOutput)
   end
end

function SpatialConvolutionMM:accGradParameters(input, gradOutput, scale)
   input, gradOutput = makeContiguous(self, input, gradOutput)
   return input.nn.SpatialConvolutionMM_accGradParameters(self, input, gradOutput, scale)
end

function SpatialConvolutionMM:type(type)
   self.finput = torch.Tensor()
   self.fgradInput = torch.Tensor()
   return parent.type(self,type)
end

function SpatialConvolutionMM:__tostring__()
   local s = string.format('%s(%d -> %d, %dx%d', torch.type(self),
         self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
   if self.dW ~= 1 or self.dH ~= 1 or self.padW ~= 0 or self.padH ~= 0 then
     s = s .. string.format(', %d,%d', self.dW, self.dH)
   end
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
     s = s .. ', ' .. self.padW .. ',' .. self.padH
   end
   return s .. ')'
end
