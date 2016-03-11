local VolumetricConvolution, parent = torch.class('nn.VolumetricConvolution', 'nn.Module')

function VolumetricConvolution:__init(nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH, padT, padW, padH)
   parent.__init(self)

   dT = dT or 1
   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kT = kT
   self.kW = kW
   self.kH = kH
   self.dT = dT
   self.dW = dW
   self.dH = dH
   self.padT = padT or 0
   self.padW = padW or self.padT
   self.padH = padH or self.padW

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kT, kH, kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kT, kH, kW)
   self.gradBias = torch.Tensor(nOutputPlane)
   self:reset()
end

function VolumetricConvolution:reset(stdv)
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

-- function to re-view the weight layout in a way that would make the MM ops happy
local function viewWeight(self)
   self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane * self.kT * self.kH * self.kW)
   if self.gradWeight and self.gradWeight:dim() > 0 then
      self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane * self.kT * self.kH * self.kW)
   end
end

local function unviewWeight(self)
   self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane, self.kT, self.kH, self.kW)
   if self.gradWeight and self.gradWeight:dim() > 0 then
      self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane, self.kT, self.kH, self.kW)
   end
end

function VolumetricConvolution:updateOutput(input)
   self.finput = self.finput or input.new()
   self.fgradInput = self.fgradInput or input.new()
   if input:type() == 'torch.CudaTensor' then
      input.THNN.VolumetricConvolution_updateOutput(
        input:cdata(),
        self.output:cdata(),
        self.weight:cdata(),
        self.bias:cdata(),
        self.finput:cdata(),
        self.fgradInput:cdata(),
        self.dT, self.dW, self.dH,
        self.padT, self.padW, self.padH
      )
   else
      viewWeight(self)
      input = makeContiguous(self, input)
      input.THNN.VolumetricConvolutionMM_updateOutput(
         input:cdata(),
         self.output:cdata(),
         self.weight:cdata(),
         self.bias:cdata(),
         self.finput:cdata(),
         self.kT, self.kW, self.kH,
         self.dT, self.dW, self.dH,
         self.padT, self.padW, self.padH
      )
      unviewWeight(self)
   end
   return self.output
end

function VolumetricConvolution:updateGradInput(input, gradOutput)
   if input:type() == 'torch.CudaTensor' then
      input.THNN.VolumetricConvolution_updateGradInput(
         input:cdata(),
         gradOutput:cdata(),
         self.gradInput:cdata(),
         self.weight:cdata(),
         self.finput:cdata(),
         self.dT, self.dW, self.dH,
         self.padT, self.padW, self.padH
      )
      return self.gradInput
   else
      if self.gradInput then
         viewWeight(self)
         input, gradOutput = makeContiguous(self, input, gradOutput)
         input.THNN.VolumetricConvolutionMM_updateGradInput(
            input:cdata(),
            gradOutput:cdata(),
            self.gradInput:cdata(),
            self.weight:cdata(),
            self.finput:cdata(),
            self.fgradInput:cdata(),
            self.kT, self.kW, self.kH,
            self.dT, self.dW, self.dH,
            self.padT, self.padW, self.padH
         )
         unviewWeight(self)
         return self.gradInput
      end
   end
end

function VolumetricConvolution:accGradParameters(input, gradOutput, scale)
   if input:type() == 'torch.CudaTensor' then
      input.THNN.VolumetricConvolution_accGradParameters(
         input:cdata(),
         gradOutput:cdata(),
         self.gradWeight:cdata(),
         self.gradBias:cdata(),
         self.finput:cdata(),
         self.fgradInput:cdata(),
         self.dT, self.dW, self.dH,
         self.padT, self.padW, self.padH,
         scale or 1
      )
   else
      input, gradOutput = makeContiguous(self, input, gradOutput)
      viewWeight(self)
      input.THNN.VolumetricConvolutionMM_accGradParameters(
         input:cdata(),
         gradOutput:cdata(),
         self.gradWeight:cdata(),
         self.gradBias:cdata(),
         self.finput:cdata(),
         scale or 1
      )
      unviewWeight(self)
   end
end

function VolumetricConvolution:type(type, tensorCache)
   if self.finput then self.finput:set() end
   if self.fgradInput then self.fgradInput:set() end
   return parent.type(self, type, tensorCache)
end

function VolumetricConvolution:clearState()
   nn.utils.clear(self, 'finput', 'fgradInput', '_input', '_gradOutput')
   return parent.clearState(self)
end

function VolumetricConvolution:__tostring__()
   local s = string.format('%s(%d -> %d, %dx%dx%d', torch.type(self),
         self.nInputPlane, self.nOutputPlane, self.kT, self.kW, self.kH)
   if self.dT ~= 1 or self.dW ~= 1 or self.dH ~= 1 or
      self.padT ~= 0 or self.padW ~= 0 or self.padH ~= 0 then
     s = s .. string.format(', %d,%d,%d', self.dT, self.dW, self.dH)
   end
   if (self.padT or self.padW or self.padH) and
      (self.padT ~=0 or self.padW ~= 0 or self.padH ~= 0) then
     s = s .. ', ' .. self.padT .. ',' .. self.padW .. ',' .. self.padH
   end
   return s .. ')'
end
