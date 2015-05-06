local FasterLookup, parent = torch.class('nn.FasterLookup', 'nn.Module')

function FasterLookup:__init(
  nIndex, dim, skipBoundChecking, scaleGradByFreq, concurrentUpdates)
  parent.__init(self)

  self.count = torch.IntTensor(nIndex);
  self.weight = torch.Tensor(nIndex, dim)
  self.gradWeight = torch.Tensor() -- do not set size yet to save mem
  self.weight:normal(0, 1.0)

  self.skipBoundChecking = skipBoundChecking and true or false
  self.scaleGradByFreq = scaleGradByFreq and true or false
  self.concUpdates = concurrentUpdates and true or false
end

function FasterLookup:type(type)
  self.weight:type(type)
  self.gradWeight:type(type)
  self.output:type(type)
end

function FasterLookup:updateOutput(input)
  local updateOutput = self.weight.nn.FasterLookup_updateOutput
  return updateOutput(self, input)
end

function FasterLookup:zeroGradParameters()
  self.gradWeight:resizeAs(self.weight)
  self.gradWeight:zero()
  self.count:zero()
end

function FasterLookup:accGradParameters(input, gradOutput, scale)
  local scale = scale or 1
  local acc = self.weight.nn.FasterLookup_accGradParameters
  acc(self, input, gradOutput, scale)
end

function FasterLookup:updateParameters(lr)
  local updateParameters = self.weight.nn.FasterLookup_updateParameters
  updateParameters(self, lr)
end

function FasterLookup:accUpdateGradParameters(input, gradOutput, lr)
  local acc = self.weight.nn.FasterLookup_accUpdateGradParameters
  acc(self, input, gradOutput, lr)
end
