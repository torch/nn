local MaskedSelect, parent = torch.class('nnd.MaskedSelect', 'nn.Module')

--[[ Sets the provided mask value for the module. ]]
function MaskedSelect:__init()
  parent.__init(self)
  self._maskIndices = torch.Tensor()
  self._buffer = torch.Tensor()
  self._gradMask = torch.Tensor()
end

--[[ Performs maskedSelect operation. ]]
function MaskedSelect:updateOutput(input)
  local input, mask = unpack(input)
  self._maskIndices:range(1, mask:nElement()):resize(mask:size())
  self._maskIndices = self._maskIndices:maskedSelect(mask):long()
  self.output = input:maskedSelect(mask)
  return self.output
end

--[[ Reverse maps unmasked gradOutput back to gradInput. ]]
function MaskedSelect:updateGradInput(input, gradOutput)
  local input, mask = unpack(input)
  self._buffer:resize(input:nElement()):zero()
  self._buffer:scatter(1, self._maskIndices, gradOutput)
  self._buffer:resize(input:size())
  self.gradInput = {self._buffer, self._gradMask:resize(mask:size()):fill(0)}
  return self.gradInput
end

function MaskedSelect:accGradParameters(input, gradOutput, scale)
end

function MaskedSelect:parameters()
  return nil
end

function MaskedSelect:type(type)
  local maskIndices = self._maskIndices
  local gradMask = self._gradMask
  local buffer = self._buffer
  if type ~= 'torch.CudaTensor' then
    self._maskIndices = nil
    self._gradMask = nil
    self._gradMask = nil
  end

  parent.type(self, type)

  if type ~= 'torch.CudaTensor' then
    self._maskIndices = maskIndices:long()
    self._buffer = buffer:double()
    self._gradMask = gradMask:byte()
  end
  return self
end

function MaskedSelect:clearStates()
  nn.utils.clear(self, {'output',
                        'gradInput',
                        '_maskIndices',
                        '_buffer',
                        '_gradMask'})
end
