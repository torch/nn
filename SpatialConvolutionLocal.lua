local SpatialConvolutionLocal, parent = torch.class('nn.SpatialConvolutionLocal', 'nn.Module')

function SpatialConvolutionLocal:__init(nInputPlane, nOutputPlane, iW, iH ,kW, kH, dW, dH, padW, padH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.iW = iW
   self.iH = iH

   self.dW = dW
   self.dH = dH
   self.padW = padW or 0
   self.padH = padH or self.padW
   self.oW = math.floor((self.padW * 2 + iW - self.kW) / self.dW) + 1
   self.oH = math.floor((self.padH * 2 + iH - self.kH) / self.dH) + 1
   assert(1 <= self.oW and 1 <= self.oH, 'illegal configuration: output width or height less than 1')

   self.weight = torch.Tensor(self.oH, self.oW, nOutputPlane, nInputPlane, kH, kW)
   self.bias = torch.Tensor(nOutputPlane, self.oH, self.oW)
   self.gradWeight = torch.Tensor():resizeAs(self.weight)
   self.gradBias = torch.Tensor():resizeAs(self.bias)

   self.finput = torch.Tensor()
   self.fgradInput = torch.Tensor()

   self:reset()
end

function SpatialConvolutionLocal:reset(stdv)
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

local function viewWeight(self)
   self.weight = self.weight:view(self.oH * self.oW, self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
   if self.gradWeight and self.gradWeight:dim() > 0 then 
      self.gradWeight = self.gradWeight:view(self.oH * self.oW, self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
   end
end

local function unviewWeight(self)
   self.weight = self.weight:view(self.oH, self.oW, self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   if self.gradWeight and self.gradWeight:dim() > 0 then 
      self.gradWeight = self.gradWeight:view(self.oH, self.oW, self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
end

local function checkInputSize(self, input)
   if input:nDimension() == 3 then
      if input:size(1) ~= self.nInputPlane or input:size(2) ~= self.iH or input:size(3) ~= self.iW then
         error(string.format('Given input size: (%dx%dx%d) inconsistent with expected input size: (%dx%dx%d).', 
                             input:size(1), input:size(2), input:size(3), self.nInputPlane, self.iH, self.iW))
      end
   elseif input:nDimension() == 4 then
      if input:size(2) ~= self.nInputPlane or input:size(3) ~= self.iH or input:size(4) ~= self.iW then
         error(string.format('Given input size: (%dx%dx%dx%d) inconsistent with expected input size: (batchsize x%dx%dx%d).', 
                              input:size(1), input:size(2), input:size(3), input:size(4), self.nInputPlane, self.iH, self.iW))
      end
   else
      error('3D or 4D(batch mode) tensor expected')
   end
end

local function checkOutputSize(self, input, output)
   if output:nDimension() ~= input:nDimension() then
      error('inconsistent dimension between output and input.')
   end
   if output:nDimension() == 3 then
      if output:size(1) ~= self.nOutputPlane or output:size(2) ~= self.oH or output:size(3) ~= self.oW then
         error(string.format('Given output size: (%dx%dx%d) inconsistent with expected output size: (%dx%dx%d).', 
                             output:size(1), output:size(2), output:size(3), self.nOutputPlane, self.oH, self.oW))
      end
   elseif output:nDimension() == 4 then
      if output:size(2) ~= self.nOutputPlane or output:size(3) ~= self.oH or output:size(4) ~= self.oW then
         error(string.format('Given output size: (%dx%dx%dx%d) inconsistent with expected output size: (batchsize x%dx%dx%d).', 
                              output:size(1), output:size(2), output:size(3), output:size(4), self.nOutputPlane, self.oH, self.oW))
      end
   else
      error('3D or 4D(batch mode) tensor expected')
   end
end

function SpatialConvolutionLocal:updateOutput(input)
   checkInputSize(self, input)
   viewWeight(self)
   input = makeContiguous(self, input)
   local out = input.nn.SpatialConvolutionLocal_updateOutput(self, input)
   unviewWeight(self)
   return out
end

function SpatialConvolutionLocal:updateGradInput(input, gradOutput)
   checkInputSize(self, input)
   checkOutputSize(self, input, gradOutput)
   if self.gradInput then
      viewWeight(self)
      input, gradOutput = makeContiguous(self, input, gradOutput)
      local out = input.nn.SpatialConvolutionLocal_updateGradInput(self, input, gradOutput)
      unviewWeight(self)
      return out
   end
end

function SpatialConvolutionLocal:accGradParameters(input, gradOutput, scale)
   checkInputSize(self, input)
   checkOutputSize(self, input, gradOutput)
   input, gradOutput = makeContiguous(self, input, gradOutput)
   viewWeight(self)
   local out = input.nn.SpatialConvolutionLocal_accGradParameters(self, input, gradOutput, scale)
   unviewWeight(self)
   return out
end

function SpatialConvolutionLocal:type(type,tensorCache)
   self.finput = torch.Tensor()
   self.fgradInput = torch.Tensor()
   return parent.type(self,type,tensorCache)
end

function SpatialConvolutionLocal:__tostring__()
   local s = string.format('%s(%d -> %d, %dx%d, %dx%d', torch.type(self),
         self.nInputPlane, self.nOutputPlane, self.iW, self.iH, self.kW, self.kH)
   if self.dW ~= 1 or self.dH ~= 1 or self.padW ~= 0 or self.padH ~= 0 then
     s = s .. string.format(', %d,%d', self.dW, self.dH)
   end
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
     s = s .. ', ' .. self.padW .. ',' .. self.padH
   end
   return s .. ')'
end
