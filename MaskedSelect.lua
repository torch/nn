local MaskedSelect, parent = torch.class('nnd.MaskedSelect', 'nn.Module')

--[[ Sets the provided mask value for the module. ]]
function MaskedSelect:__init()
  -- mask must be a tensor, convert to byte tensor
  parent.__init(self)
end

--[[ Performs maskedSelect operation. ]]
function MaskedSelect:updateOutput(input)
  local input, mask = unpack(input)
  self.output = input:maskedSelect(mask)
  return self.output
end

--[[ Reverse maps unmasked gradOutput back to gradInput. ]]
function MaskedSelect:updateGradInput(input, gradOutput)
  local input, mask = unpack(input)
  local maskIndices =
    torch.range(1, mask:nElement()):resizeAs(mask:double()):maskedSelect(mask)
  local gradInput = torch.Tensor(input:nElement()):zero()
  gradInput:scatter(1, maskIndices:long(), gradOutput)
  gradInput:resizeAs(input:double())
  self.gradInput = {gradInput, torch.Tensor():resizeAs(input):fill(0):byte()}
  return self.gradInput
end

function MaskedSelect:accGradParameters(input, gradOutput, scale)
end

function MaskedSelect:parameters()
  return nil
end

function MaskedSelect:type(type)
  return self
end

function MaskedSelect:clearStates()
  nn.utils.clear(self, {'output', 'gradInput'})
end
