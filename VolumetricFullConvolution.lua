local VolumetricFullConvolution, parent = torch.class('nn.VolumetricFullConvolution','nn.Module')

function VolumetricFullConvolution:__init(nInputPlane, nOutputPlane,
                                          kT, kW, kH,         -- kernel size
                                          dT, dW, dH,         -- stride
                                          padT, padW, padH,   -- padding
                                          adjT, adjW, adjH)   -- extra output adjustment
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1
   dT = dT or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.kT = kT
   self.dW = dW
   self.dH = dH
   self.dT = dT
   self.padW = padW or 0
   self.padH = padH or 0
   self.padT = padT or 0
   self.adjW = adjW or 0
   self.adjH = adjH or 0
   self.adjT = adjT or 0

   if self.adjW > self.dW - 1 or self.adjH > self.dH - 1 or self.adjT > self.dT - 1 then
      error('adjW, adjH and adjT must be smaller than self.dW - 1,' ..
            ' self.dH - 1 and self.dT - 1 respectively')
   end

   self.weight = torch.Tensor(nInputPlane, nOutputPlane, kT, kH, kW)
   self.gradWeight = torch.Tensor(nInputPlane, nOutputPlane, kT, kH, kW)
   self.bias = torch.Tensor(self.nOutputPlane)
   self.gradBias = torch.Tensor(self.nOutputPlane)

   self.finput = torch.Tensor()
   self.fgradInput = torch.Tensor()

   self:reset()
end

function VolumetricFullConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      local nInputPlane = self.nInputPlane
      local kT = self.kT
      local kH = self.kH
      local kW = self.kW
      stdv = 1/math.sqrt(kW*kH*kT*nInputPlane)
   end
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv)
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

function VolumetricFullConvolution:backCompatibility()
   -- Transpose the weight when loading from an old version
   if not self.adjW then
      self.weight = self.weight:transpose(1, 2):contiguous()
   end

   -- Rename the padding when loading from an old version
   self.padW = self.padW or self.pW
   self.padH = self.padH or self.pH
   self.padT = self.padT or self.pT

   self.adjW = self.adjW or 0
   self.adjH = self.adjH or 0
   self.adjT = self.adjT or 0
end

function VolumetricFullConvolution:updateOutput(input)
   self:backCompatibility()

   input = makeContiguous(self, input)
   input.THNN.VolumetricFullConvolution_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.weight:cdata(),
      self.bias:cdata(),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.dT, self.dW, self.dH,
      self.padT, self.padW, self.padH,
      self.adjT, self.adjW, self.adjH
   )

   return self.output
end

function VolumetricFullConvolution:updateGradInput(input, gradOutput)
   self:backCompatibility()

   input, gradOutput = makeContiguous(self, input, gradOutput)
   input.THNN.VolumetricFullConvolution_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.weight:cdata(),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.dT, self.dW, self.dH,
      self.padT, self.padW, self.padH,
      self.adjT, self.adjW, self.adjH
   )
   return self.gradInput
end

function VolumetricFullConvolution:accGradParameters(input, gradOutput, scale)
   self:backCompatibility()

   input, gradOutput = makeContiguous(self, input, gradOutput)
   input.THNN.VolumetricFullConvolution_accGradParameters(
      input:cdata(),
      gradOutput:cdata(),
      self.gradWeight:cdata(),
      self.gradBias:cdata(),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.dT, self.dW, self.dH,
      self.padT, self.padW, self.padH,
      self.adjT, self.adjW, self.adjH,
      scale or 1
   )
end

function VolumetricFullConvolution:type(type, tensorCache)
   self.finput = torch.Tensor()
   self.fgradInput = torch.Tensor()
   return parent.type(self, type, tensorCache)
end

function VolumetricFullConvolution:__tostring__()
   local s = string.format('%s(%d -> %d, %dx%dx%d', torch.type(self),
   self.nInputPlane, self.nOutputPlane, self.kT, self.kW, self.kH)
   if self.dT ~= 1 or self.dW ~= 1 or self.dH ~= 1 or self.padT ~= 0 or self.padW ~= 0 or self.padH ~= 0 then
      s = s .. string.format(', %d,%d,%d', self.dT, self.dW, self.dH)
   end
   if (self.padT or self.padW or self.padH) and (self.padT ~= 0 or self.padW ~= 0 or self.padH ~= 0) then
      s = s .. ', ' .. self.padT .. ',' .. self.padW .. ',' .. self.padH
   end
   if (self.adjT or self.adjW or self.adjH) and (self.adjT ~= 0 or self.adjW ~= 0 or self.adjH ~= 0) then
      s = s .. ', ' .. self.adjT .. ',' .. self.adjW .. ',' .. self.adjH
   end
   return s .. ')'
end
