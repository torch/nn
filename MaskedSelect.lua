local MaskedSelect, parent = torch.class('nnd.MaskedSelect', 'nn.Module')

--[[ Sets the provided mask value for the module. ]]
function MaskedSelect:__init()
  -- mask must be a tensor, convert to byte tensor
  parent.__init(self)
end

--[[ Performs maskedSelect operation. ]]
function MaskedSelect:updateOutput(input)
  local input, mask = unpack(input)
  self.output:maskedSelect(input, mask)
  self._maskIndices =
    torch.range(1, mask:nElement()):resize(mask:size()):maskedSelect(mask)
  return self.output
end

--[[ Reverse maps unmasked gradOutput back to gradInput. ]]
function MaskedSelect:updateGradInput(input, gradOutput)
  local input, mask = unpack(input)
  local gradInput = torch.Tensor(input:nElement()):zero()
  gradInput:scatter(1, self._maskIndices:long(), gradOutput)
  gradInput:resize(input:size())
  self.gradInput = {gradInput, torch.Tensor():resize(mask:size()):fill(0)}
  return self.gradInput
end

function MaskedSelect:accGradParameters(input, gradOutput, scale)
end

function MaskedSelect:parameters()
  return nil
end

function MaskedSelect:type(type)
  local maskIndices = self._maskIndices
  if type ~= 'torch.CudaTensor' then
    -- ByteTensors must remain ByteTensors
    self._maskIndices = nil
  end
  parent.type(self, type)
  if type ~= 'torch.CudaTensor' then
    self._maskIndices = maskIndices:byte()
  end
  return self
end

function MaskedSelect:clearStates()
  nn.utils.clear(self, {'output', 'gradInput', '_maskIndices'})
end
