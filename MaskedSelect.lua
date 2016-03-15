local MaskedSelect, parent = torch.class('nn.MaskedSelect', 'nn.Module')

--[[ Sets the provided mask value for the module. ]]
function MaskedSelect:__init(mask)
  -- mask must be a tensor, convert to byte tensor
  parent.__init(self)
  self.mask = mask
  self.maskIndices =
    torch.range(1, mask:nElement()):resizeAs(mask:double()):maskedSelect(mask)
end

--[[ Performs maskedSelect operation. ]]
function MaskedSelect:updateOutput(input)
  self.output = input:maskedSelect(self.mask)
  return self.output
end

--[[ Reverse maps unmasked gradOutput back to gradInput. ]]
function MaskedSelect:updateGradInput(input, gradOutput)
  self.gradInput = torch.Tensor(input:nElement()):zero()
  self.gradInput:scatter(1, self.maskIndices:long(), gradOutput)
  self.gradInput:resizeAs(input:double())
  return self.gradInput
end

function MaskedSelect:accGradParameters(input, gradOutput, scale)
end

function MaskedSelect:parameters()
  return nil
end

function MaskedSelect:type(type)
  local mask = self.mask
  local maskIndices = self.maskIndices
  if type ~= 'torch.CudaTensor' then
    -- ByteTensors must remain ByteTensors
    self.mask = nil
    self.maskIndices = nil
  end
  parent.type(self, type)
  if type ~= 'torch.CudaTensor' then
    self.mask = mask:byte()
    self.maskIndices = maskIndices:byte()
  end
  return self
end

function MaskedSelect:clearStates()
  nn.utils.clear(self, {'output', 'gradInput'})
end
