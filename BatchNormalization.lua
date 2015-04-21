--[[
   This file implements Batch Normalization as described in the paper:
   "Batch Normalization: Accelerating Deep Network Training
                         by Reducing Internal Covariate Shift"
                   by Sergey Ioffe, Christian Szegedy

   This implementation is useful for inputs NOT coming from convolution layers.
   For Convolution layers, see SpatialBatchNormalization.lua

   The operation implemented is:
   y =     ( x - mean(x) )
        -------------------- * gamma + beta
       standard-deviation(x)
   where gamma and beta are learnable parameters.

   The learning of gamma and beta is optional.

   Usage:
   with    learnable parameters: nn.BatchNormalization(N [, eps] [,momentum])
                                 where N = dimensionality of input
   without learnable parameters: nn.BatchNormalization(0 [, eps] [,momentum])

   eps is a small value added to the standard-deviation to avoid divide-by-zero.
       Defaults to 1e-5

   In training time, this layer keeps a running estimate of it's computed mean and std.
   The running sum is kept with a default momentup of 0.1 (unless over-ridden)
   In test time, this running mean/std is used to normalize.
]]--
local BN,parent = torch.class('nn.BatchNormalization', 'nn.Module')

function BN:__init(nOutput, eps, momentum)
   parent.__init(self)
   assert(nOutput and type(nOutput) == 'number',
          'Missing argument #1: dimensionality of input. ' ..
          'Give 0 for no affine transform')
   self.eps = eps or 1e-5
   self.train = true
   self.momentum = momentum or 0.1
   self.running_mean = torch.Tensor()
   self.running_std = torch.Tensor()

   if nOutput > 0 then self.affine = true end
   if self.affine then
      self.weight = torch.Tensor(nOutput)
      self.bias = torch.Tensor(nOutput)
      self.gradWeight = torch.Tensor(nOutput)
      self.gradBias = torch.Tensor(nOutput)
      self:reset()
   end
end

function BN:reset()
   self.weight:uniform()
   self.bias:zero()
end

function BN:updateOutput(input)
   assert(input:dim() == 2, 'only mini-batch supported (2D tensor), got '
             .. input:dim() .. 'D tensor instead')
   local nBatch = input:size(1)

   -- buffers that are reused
   self.buffer = self.buffer or input.new()
   self.buffer2 = self.buffer2 or input.new()
   self.centered = self.centered or input.new()
   self.centered:resizeAs(input)
   self.std = self.std or input.new()
   self.normalized = self.normalized or input.new()
   self.normalized:resizeAs(input)
   self.output:resizeAs(input)
   self.gradInput:resizeAs(input)
   if self.train == false then
      assert(self.running_mean:nDimension() ~= 0,
             'Module never run on training data. First run on some training data before evaluating.')
      self.output:copy(input)
      self.buffer:repeatTensor(self.running_mean, nBatch, 1)
      self.output:add(-1, self.buffer)
      self.buffer:repeatTensor(self.running_std, nBatch, 1)
      self.output:cmul(self.buffer)
   else -- training mode
      if self.running_mean:nDimension() == 0 then
         self.running_mean:resize(input:size(2)):zero()
      end
      if self.running_std:nDimension() == 0 then
         self.running_std:resize(input:size(2)):zero()
      end
      -- calculate mean over mini-batch
      self.buffer:mean(input, 1)                        -- E(x) = expectation of x.
      self.running_mean:mul(1 - self.momentum):add(self.momentum, self.buffer) -- add to running mean
      self.buffer:repeatTensor(self.buffer, nBatch, 1)

      -- subtract mean
      self.centered:add(input, -1, self.buffer)         -- x - E(x)

      -- calculate standard deviation over mini-batch
      self.buffer:copy(self.centered):cmul(self.buffer) -- [x - E(x)]^2

      -- 1 / E([x - E(x)]^2)
      self.std:mean(self.buffer, 1):add(self.eps):sqrt():pow(-1)
      self.running_std:mul(1 - self.momentum):add(self.momentum, self.std) -- add to running stdv
      self.buffer:repeatTensor(self.std, nBatch, 1)

      -- divide standard-deviation + eps
      self.output:cmul(self.centered, self.buffer)
      self.normalized:copy(self.output)
   end

   if self.affine then
      -- multiply with gamma and add beta
      self.buffer:repeatTensor(self.weight, nBatch, 1)
      self.output:cmul(self.buffer)
      self.buffer:repeatTensor(self.bias, nBatch, 1)
      self.output:add(self.buffer)
   end
   return self.output
end

function BN:updateGradInput(input, gradOutput)
   assert(input:dim() == 2, 'only mini-batch supported')
   assert(gradOutput:dim() == 2, 'only mini-batch supported')
   assert(self.train == true, 'should be in training mode when self.train is true')
   local nBatch = input:size(1)

   self.gradInput:cmul(self.centered, gradOutput)
   self.buffer:mean(self.gradInput, 1)
   self.gradInput:repeatTensor(self.buffer, nBatch, 1)
   self.gradInput:cmul(self.centered):mul(-1)
   self.buffer:repeatTensor(self.std, nBatch, 1)
   self.gradInput:cmul(self.buffer):cmul(self.buffer)

   self.buffer:mean(gradOutput, 1)
   self.buffer:repeatTensor(self.buffer, nBatch, 1)
   self.gradInput:add(gradOutput):add(-1, self.buffer)
   self.buffer:repeatTensor(self.std, nBatch, 1)
   self.gradInput:cmul(self.buffer)

   if self.affine then
      self.buffer:repeatTensor(self.weight, nBatch, 1)
      self.gradInput:cmul(self.buffer)
   end

   return self.gradInput
end

function BN:accGradParameters(input, gradOutput, scale)
   if self.affine then
      scale = scale or 1.0
      self.buffer2:resizeAs(self.normalized):copy(self.normalized)
      self.buffer2:cmul(gradOutput)
      self.buffer:sum(self.buffer2, 1) -- sum over mini-batch
      self.gradWeight:add(scale, self.buffer)
      self.buffer:sum(gradOutput, 1) -- sum over mini-batch
      self.gradBias:add(scale, self.buffer)
   end
end
