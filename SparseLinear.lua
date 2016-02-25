local THNN = require 'nn.THNN'
local SparseLinear, parent = torch.class('nn.SparseLinear', 'nn.Module')

function SparseLinear:__init(inputSize, outputSize)
   parent.__init(self)

   self.weightDecay = 0
   self.weight = torch.Tensor(outputSize, inputSize):zero()
   self.bias = torch.Tensor(outputSize):zero()
   self.gradWeight = torch.Tensor(outputSize, inputSize):zero()
   self.gradBias = torch.Tensor(outputSize):zero()
   self.lastInput = nil

   if torch.getnumthreads() > 1 and outputSize >= 128 then
      self.shardBuffer = torch.Tensor(outputSize, torch.getnumthreads())
   end

   -- state
   self.gradInput:resize(inputSize)
   self.output:resize(outputSize)

   self:reset()
end

function SparseLinear:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv):mul(0.000001)
end

function SparseLinear:reshapeInput(input)
   if input:dim() == 2 then
      return input:view(1, input:size(1), input:size(2)), false
   else
      return input, true
   end
end

function SparseLinear:updateOutput(input)
   self.cudaBuffer = self.cudaBuffer or input.new()
   local input, batchMode = self:reshapeInput(input)

   input.THNN.SparseLinear_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.weight:cdata(),
      self.bias:cdata(),
      self.cudaBuffer:cdata(),
      THNN.optionalTensor(self.shardBuffer)
   )

   -- fix output size for batchSize = 1
   if not batchMode then
      self.output:set(self.output:view(self.output:size(2)))
   end

   return self.output
end

function SparseLinear:accGradParameters(input, gradOutput, scale)
   local input, batchMode = self:reshapeInput(input)

   self.lastInput = self.lastInput or input.new()
   self.lastInput:resizeAs(input):copy(input)
   if not batchMode then
      gradOutput = gradOutput:view(1, gradOutput:size(1))
   end

   input.THNN.SparseLinear_accGradParameters(
      input:cdata(),
      gradOutput:cdata(),
      self.gradWeight:cdata(),
      self.gradBias:cdata(),
      self.weight:cdata(),
      self.bias:cdata(),
      self.weightDecay or 0,
      scale or 1
   )
end

function SparseLinear:updateGradInput(input, gradOutput)
   if self.gradInput then
      local input, batchMode = self:reshapeInput(input)
      if not batchMode then
         gradOutput = gradOutput:view(1, gradOutput:size(1))
      end
      input.THNN.SparseLinear_updateGradInput(
         input:cdata(),
         gradOutput:cdata(),
         self.gradInput:cdata(),
         self.weight:cdata()
      )
      -- fix gradInput size for batchSize = 1
      if not batchMode then
         self.gradInput:set(self.gradInput:view(self.gradInput:size(2), self.gradInput:size(3)))
      end

      return self.gradInput
   end
end

function SparseLinear:updateParameters(learningRate)
   if self.lastInput then
      self.lastInput.THNN.SparseLinear_updateParameters(
         self.weight:cdata(),
         self.bias:cdata(),
         self.gradWeight:cdata(),
         self.gradBias:cdata(),
         self.lastInput:cdata(),
         learningRate
      )
   else
      parent.updateParameters(self, learningRate)
   end
end

function SparseLinear:zeroGradParameters()
   if self.lastInput then
      self.lastInput.THNN.SparseLinear_zeroGradParameters(
         self.gradWeight:cdata(),
         self.gradBias:cdata(),
         self.lastInput:cdata()
      )
   else
      parent.zeroGradParameters(self)
   end
end

function SparseLinear:clearState()
   if self.lastInput then self.lastInput:set() end
   if self.cudaBuffer then self.cudaBuffer:set() end
   return parent.clearState(self)
end
