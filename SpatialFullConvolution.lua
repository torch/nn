local SpatialFullConvolution, parent = torch.class('nn.SpatialFullConvolution','nn.Module')

function SpatialFullConvolution:__init(nInputPlane, nOutputPlane,
                                       kW, kH, dW, dH, padW, padH, adjW, adjH)
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
   self.padH = padH or 0
   self.adjW = adjW or 0
   self.adjH = adjH or 0

   if self.adjW > self.dW - 1 or self.adjH > self.dH - 1 then
      error('adjW and adjH must be smaller than self.dW - 1' ..
            ' and self.dH - 1 respectively')
   end

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

function SpatialFullConvolution:backCompatibility()
  self.adjW = self.adjW or 0
  self.adjH = self.adjH or 0
end

function SpatialFullConvolution:updateOutput(input)
  self.finput = self.finput or input.new()
  self.fgradInput = self.fgradInput or input.new()
  self:backCompatibility()

  input = makeContiguous(self, input)
  input.THNN.SpatialFullConvolution_updateOutput(
    input:cdata(),
    self.output:cdata(),
    self.weight:cdata(),
    self.bias:cdata(),
    self.finput:cdata(),
    self.fgradInput:cdata(),
    self.kW, self.kH,
    self.dW, self.dH,
    self.padW, self.padH,
    self.adjW, self.adjH
  )

  return self.output
end

function SpatialFullConvolution:updateGradInput(input, gradOutput)
  self:backCompatibility()

  if self.gradInput then
    input, gradOutput = makeContiguous(self, input, gradOutput)
    input.THNN.SpatialFullConvolution_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.weight:cdata(),
      self.finput:cdata(),
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH,
      self.adjW, self.adjH
    )

    return self.gradInput
  end
end

function SpatialFullConvolution:accGradParameters(input, gradOutput, scale)
  scale = scale or 1
  self:backCompatibility()

  input, gradOutput = makeContiguous(self, input, gradOutput)
  input.THNN.SpatialFullConvolution_accGradParameters(
    input:cdata(),
    gradOutput:cdata(),
    self.gradWeight:cdata(),
    self.gradBias:cdata(),
    self.finput:cdata(),
    self.fgradInput:cdata(),
    self.kW, self.kH,
    self.dW, self.dH,
    self.padW, self.padH,
    self.adjW, self.adjH,
    scale
  )
end

function SpatialFullConvolution:type(type, tensorCache)
  self.finput = self.finput and torch.Tensor()
  self.fgradInput = self.fgradInput and torch.Tensor()
  return parent.type(self, type, tensorCache)
end

function SpatialFullConvolution:__tostring__()
  local s = string.format('%s(%d -> %d, %dx%d', torch.type(self),
  self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
  if self.dW ~= 1 or self.dH ~= 1 or self.padW ~= 0 or self.padH ~= 0 then
    s = s .. string.format(', %d,%d', self.dW, self.dH)
  end
  if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
    s = s .. ', ' .. self.padW .. ',' .. self.padH
  end
  if (self.adjW or self.adjH) and (self.adjW ~= 0 or self.adjH ~= 0) then
    s = s .. ', ' .. self.adjW .. ',' .. self.adjH
  end
  return s .. ')'
end

function SpatialFullConvolution:clearState()
   nn.utils.clear(self, 'finput', 'fgradInput', '_input', '_gradOutput')
   return parent.clearState(self)
end

