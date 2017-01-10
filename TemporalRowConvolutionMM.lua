local THNN = require "nn.THNN"

local TemporalRowConvolutionMM, parent = torch.class("nn.TemporalRowConvolutionMM", "nn.Module")

function TemporalRowConvolutionMM:__init(inputFrameSize, kW, dW)
  parent.__init(self)

  self.inputFrameSize = inputFrameSize
  self.kW = kW
  self.dW = dW or 1

  self.weight = torch.Tensor(inputFrameSize, kW)
  self.bias = torch.Tensor(inputFrameSize)
  self.gradWeight = torch.Tensor(inputFrameSize, kW)
  self.gradBias = torch.Tensor(inputFrameSize)

  self:reset()
end

function TemporalRowConvolutionMM:noBias()
  self.bias = nil
  self.gradBias = nil
  return self
end

function TemporalRowConvolutionMM:reset(stdv)
  if stdv then
    stdv = stdv * math.sqrt(3)
  else
    stdv = 1 / math.sqrt(self.kW * self.inputFrameSize)
  end
  if nn.oldseed then
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

function TemporalRowConvolutionMM:updateOutput(input)
  assert(input.THNN, torch.type(input)..".THNN backend not imported")
  self.finput = self.finput or input.new()
  self.fgradInput = self.fgradInput or input.new()

  input.THNN.TemporalRowConvolutionMM_updateOutput(
    input:cdata(),
    self.output:cdata(),
    self.weight:cdata(),
    THNN.optionalTensor(self.bias),
    self.finput:cdata(),
    self.fgradInput:cdata(),
    self.kW,
    self.dW,
    0 -- would be self.padW
  )

  return self.output
end

function TemporalRowConvolutionMM:updateGradInput(input, gradOutput)
  assert(input.THNN, torch.type(input)..".THNN backend not imported")

  if self.gradInput then
    input.THNN.TemporalRowConvolutionMM_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.weight:cdata(),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.kW,
      self.dW,
      0 -- would be self.padW
    )
    return self.gradInput
  end
end

function TemporalRowConvolutionMM:accGradParameters(input, gradOutput, scale)
  assert(input.THNN, torch.type(input)..".THNN backend not imported")

  input.THNN.TemporalRowConvolutionMM_accGradParameters(
    input:cdata(),
    gradOutput:cdata(),
    self.gradWeight:cdata(),
    THNN.optionalTensor(self.gradBias),
    self.finput:cdata(),
    self.fgradInput:cdata(),
    self.kW,
    self.dW,
    0, -- would be self.padW
    scale or 1)
end

function TemporalRowConvolutionMM:type(type, tensorCache)
  if self.finput then self.finput:set() end
  if self.fgradInput then self.fgradInput:set() end
  return parent.type(self, type, tensorCache)
end

function TemporalRowConvolutionMM:__tostring__()
  local s = string.format("%s(%d, %d", torch.type(self), self.inputFrameSize, self.kW)
  if self.dW ~= 1 then
    s = s .. string.format(", %d", self.dW)
  end
  if self.padW and self.padW ~= 0 then -- currently padding is not supported
    s = s .. ", " .. self.padW
  end
  if self.bias then
    return s .. ")"
  else
    return s .. ") without bias"
  end
end

function TemporalRowConvolutionMM:clearState()
  nn.utils.clear(self, "finput", "fgradInput", "_input", "_gradOutput")
  return parent.clearState(self)
end
