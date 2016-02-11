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
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
         self.bias[i] = torch.uniform(-stdv, stdv) * 0.000001
      end
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv):mul(0.000001)
   end
end

function SparseLinear:updateOutput(input)
   input.THNN.SparseLinear_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.weight:cdata(),
      self.bias:cdata(),
      THNN.optionalTensor(self.shardBuffer)
   )
   return self.output
end

function SparseLinear:accGradParameters(input, gradOutput, scale)
   if not self.lastInput then
      self.lastInput = input:clone()
   else
      self.lastInput:resizeAs(input):copy(input)
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
     input.THNN.SparseLinear_updateGradInput(
        input:cdata(),
        gradOutput:cdata(),
        self.gradInput:cdata(),
        self.weight:cdata()
     )
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
   return parent.clearState(self)
end
