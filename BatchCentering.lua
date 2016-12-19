local BC,parent = torch.class('nn.BatchCentering', 'nn.Module')

BC.nDim = 2

function BC:__init(nOutput, momentum)
    parent.__init(self)
    assert(nOutput and type(nOutput) == 'number',
           'Missing argument #1: dimensionality of input. ')
    self.train = true
    self.momentum = momentum or 0.1
    self.nOutput = nOutput
    self.running_mean = torch.zeros(nOutput)
end

function BC:reset()
    self.running_mean:zero()
end

function BC:checkInputDim(input)
   local iDim = input:dim()
   assert(iDim == self.nDim or
              (iDim == self.nDim - 1 and self.train == false), string.format(
      'only mini-batch supported (%dD tensor), got %dD tensor instead',
      self.nDim, iDim))
   local featDim = (iDim == self.nDim - 1) and 1 or 2
   assert(input:size(featDim) == self.running_mean:nElement(), string.format(
      'got %d-feature tensor, expected %d',
      input:size(featDim), self.running_mean:nElement()))
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

local function makeBatch(self, input)
    local iDim = input:dim()
    if self.train == false and iDim == self.nDim - 1 then
        return nn.utils.addSingletonDimension(input, input, 1)
    else
        return input
    end
end

function BC:updateOutput(input)
   self:checkInputDim(input)
   input = makeContiguous(self, input)
   input = makeBatch(self, input)

   local nBatch, nOutput = input:size(1), self.nOutput
   local input_ = input
   input = input:view(nBatch, nOutput, -1)
    self.output:resizeAs(input)
    if self.train == false then
        self.output:copy(input)
        local meanReplicated = torch.repeatTensor(self.running_mean, nBatch, 1, input:size(3))
        self.output:add(-1, meanReplicated)
    else -- training mode
        -- calculate mean over mini-batch
        local mean = torch.mean(input, 1):mean(3)
        local meanReplicated = torch.repeatTensor(mean, nBatch, 1, input:size(3))
        self.output:add(input, -1, meanReplicated)
        self.running_mean:mul(1 - self.momentum):add(self.momentum, mean)
    end
    self.output:set(self.output:viewAs(input_))

    return self.output
end

function BC:updateGradInput(input, gradOutput)
   self:checkInputDim(input)
   self:checkInputDim(gradOutput)
   input, gradOutput = makeContiguous(self, input, gradOutput)
   input = makeBatch(self, input)
   gradOutput = makeBatch(self, gradOutput)

   local nBatch, nOutput = input:size(1), self.nOutput
   local input_ = input
   input = input:view(nBatch, nOutput, -1)
   gradOutput = gradOutput:view(nBatch, nOutput, -1)

   if self.train then
        local gmean = torch.mean(gradOutput, 1):mean(3)
        local gmeanRepeat = torch.repeatTensor(gmean, nBatch, 1, input:size(3))
        self.gradInput:resizeAs(gradOutput)
        self.gradInput:copy(gradOutput):add(-1, gmeanRepeat)
    else
        self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   end
   self.gradInput:set(self.gradInput:viewAs(input_))
   return self.gradInput
end
