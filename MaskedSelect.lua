local MaskedSelect, parent = torch.class('nnd.MaskedSelect', 'nn.Module')

--[[ Sets the provided mask value for the module. ]]
function MaskedSelect:__init()
  parent.__init(self)
  self._maskIndices = torch.Tensor()
  self._maskIndexBuffer = torch.Tensor()
  self._gradBuffer = torch.Tensor()
  self._gradMask = torch.Tensor()
end

--[[ Performs maskedSelect operation. ]]
function MaskedSelect:updateOutput(input)
  local input, mask = unpack(input)
  self.output:maskedSelect(input, mask)
  return self.output
end

--[[ Reverse maps unmasked gradOutput back to gradInput. ]]
function MaskedSelect:updateGradInput(input, gradOutput)
  local input, mask = unpack(input)
  self._maskIndexBuffer:range(1, mask:nElement()):resize(mask:size())
  self._maskIndices:maskedSelect(self._maskIndexBuffer, mask)
  self._gradBuffer:resize(input:nElement()):zero()
  self._gradBuffer:scatter(1, self._maskIndices:long(), gradOutput)
  self._gradBuffer:resize(input:size())
  self.gradInput = {self._gradBuffer, self._gradMask:resize(mask:size()):fill(0)}
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
  local gradBuffer = self._gradBuffer
  local maskIndexBuffer = self._maskIndexBuffer
  if type ~= 'torch.CudaTensor' then
    self._maskIndices = nil
    self._gradBuffer = nil
    self._maskBuffer = nil
    self._gradMask = nil
  end

  parent.type(self, type)

  if type ~= 'torch.CudaTensor' then
    self._maskIndices = maskIndices:long()
    self._gradBuffer = gradBuffer:double()
    self._maskIndexBuffer = maskIndexBuffer:double()
    self._gradMask = gradMask:byte()
  end
  return self
end

function MaskedSelect:clearStates()
  nn.utils.clear(self, {'output',
                        'gradInput',
                        '_maskIndices',
                        '_gradBuffer',
                        '_maskBuffer',
                        '_gradMask'})
end
