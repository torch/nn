--[[
   This file implements Batch Normalization as described in the paper:
   "Batch Normalization: Accelerating Deep Network Training
                         by Reducing Internal Covariate Shift"
                by Sergey Ioffe, Christian Szegedy

   This implementation is useful for inputs coming from convolution layers.
   For Non-convolutional layers, see BatchNormalization.lua

   The operation implemented is:
   y =     ( x - mean(x) )
   -------------------- * gamma + beta
   standard-deviation(x)
   where gamma and beta are learnable parameters.

   The learning of gamma and beta is optional.

   Usage:
   with    learnable parameters: nn.BatchNormalization(N [,eps] [,momentum])
                                 where N = dimensionality of input
   without learnable parameters: nn.BatchNormalization(0 [,eps] [,momentum])

   eps is a small value added to the standard-deviation to avoid divide-by-zero.
       Defaults to 1e-5

   In training time, this layer keeps a running estimate of it's computed mean and std.
   The running sum is kept with a default momentup of 0.1 (unless over-ridden)
   In test time, this running mean/std is used to normalize.

]]--
local BN,parent = torch.class('nn.SpatialBatchNormalization', 'nn.Module')

function BN:__init(nFeature, eps, momentum, affine)
   parent.__init(self)
   assert(nFeature and type(nFeature) == 'number',
          'Missing argument #1: Number of feature planes. ')
   assert(nFeature ~= 0, 'To set affine=false call SpatialBatchNormalization'
     .. '(nFeature,  eps, momentum, false) ')
   if affine ~=nil then
      assert(type(affine) == 'boolean', 'affine has to be true/false')
      self.affine = affine
   else
      self.affine = true
   end
   self.eps = eps or 1e-5
   self.train = true
   self.momentum = momentum or 0.1
   -- flag to indicate when forward pass was the most recent call
   self.forwardDone = false

   self.running_mean = torch.zeros(nFeature)
   self.running_std = torch.ones(nFeature)
   if self.affine then
      self.weight = torch.Tensor(nFeature)
      self.bias = torch.Tensor(nFeature)
      self.gradWeight = torch.Tensor(nFeature)
      self.gradBias = torch.Tensor(nFeature)
      self:reset()
   end
end

function BN:reset()
   self.weight:uniform()
   self.bias:zero()
end

-- used in tests
function BN:resetGradParams()
   self.gradWeight:zero()
   self.gradBias:zero()
end

-- used in tests
function BN:resetRunningStats(nFeature)
   self.running_mean:zeros(nFeature)
   self.running_std:ones(nFeature)
end

-- helper to determine if a value is a power of 2
local function isPowerOf2(x)
   while (((x % 2) == 0) and x > 1) do
      x = x/2
   end
   return x == 1
end

-- checks if the dimensions are powers of 2, and nBatch within range
-- to be able to apply the optimized SBN routine
local function validOptDimensions(nBatch, nFeature, nSpatial)
   local batchDimValid = (nBatch <= 32 or isPowerOf2(nBatch)) and nBatch <=1024
   return batchDimValid and isPowerOf2(nSpatial) and nFeature <= 1024
end


function BN:updateOutput(input)
   assert(input:dim() == 4, 'only mini-batch supported (4D tensor), got '
             .. input:dim() .. 'D tensor instead')
   local nBatch = input:size(1)
   local nFeature = input:size(2)
   local iH = input:size(3)
   local iW = input:size(4)
   local nSpatial = iH * iW

   -- buffers that are reused on both CPU and GPU
   self.centered = self.centered or input.new()
   self.centered:resizeAs(input)
   self.output:resizeAs(input)
   self.gradInput:resizeAs(input)
   self.buffer = self.buffer or input.new()
   self.normalized = self.normalized or input.new()
   self.normalized:resizeAs(input)
   self.forwardDone = true

   -- optimized code path
   if input:type()=='torch.CudaTensor' and validOptDimensions(nBatch, nFeature, nSpatial) then
      if self.train == false then
         -- pre-allocate buffers for transpose input and output
         self.transposedInput = self.transposedInput or input.new()
         self.transposedInput:resize(nFeature, nSpatial, nBatch)
         self.transposedOutput = self.transposedOutput or input.new()
         self.transposedOutput:resize(nFeature, nSpatial, nBatch)

         if self.affine == true then
            input.nn.SpatialBatchNormalization_forwardInferenceAffine(self, input, nBatch, nFeature, nSpatial)
         else
            input.nn.SpatialBatchNormalization_forwardInference(self, input, nBatch, nFeature, nSpatial)
         end
      else
         self.meanBuffer = self.meanBuffer or input.new()
         self.stdBuffer = self.stdBuffer or input.new()
         -- allocate these in the c function?
         self.meanBuffer:resize(nFeature)
         self.stdBuffer:resize(nFeature)
         self.batchAgg = self.batchAgg or input.new() --[nFeature x nSpatial]
         self.batchAgg:resize(nFeature, nSpatial)

         -- pre-allocate buffers for transpose input and output
         self.transposedInput = self.transposedInput or input.new()
         self.transposedInput:resize(nFeature, nSpatial, nBatch)
         self.transposedOutput = self.transposedOutput or input.new()
         self.transposedOutput:resize(nFeature, nSpatial, nBatch)

         -- calculate mean and std over mini-batch
         -- outputs the results into self.output, and mean and var from this batch into self.meanBuffer and self.stdBuffer
         if self.affine then
            input.nn.SpatialBatchNormalization_updateOutputAffine(self, input, nBatch, nFeature, nSpatial)
         else
            input.nn.SpatialBatchNormalization_updateOutput(self, input, nBatch, nFeature, nSpatial)
         end

         self.running_mean:mul(1 - self.momentum):add(self.momentum, self.meanBuffer) -- add to running mean
         self.running_std:mul(1 - self.momentum):add(self.momentum, self.stdBuffer) -- add to running stdv
      end
   else -- original path
      if self.train == false then
         self.output:copy(input)
         self.buffer:repeatTensor(self.running_mean:view(1, nFeature, 1, 1), nBatch, 1, iH, iW)
         self.output:add(-1, self.buffer)
         self.buffer:repeatTensor(self.running_std:view(1, nFeature, 1, 1), nBatch, 1, iH, iW)
         self.output:cmul(self.buffer)
      else
         -- buffers
         self.buffer2 = self.buffer2 or input.new()
         self.std = self.std or input.new()
         -- calculate mean over mini-batch, over feature-maps
         local in_folded = input:view(nBatch, nFeature, iH * iW)
         self.buffer:mean(in_folded, 1)
         self.buffer2:mean(self.buffer, 3)
         self.running_mean:mul(1 - self.momentum):add(self.momentum, self.buffer2) -- add to running mean
         self.buffer:repeatTensor(self.buffer2:view(1, nFeature, 1, 1),
                                  nBatch, 1, iH, iW)
         -- subtract mean
         self.centered:add(input, -1, self.buffer)                  -- x - E(x)
         -- calculate standard deviation over mini-batch
         self.buffer:copy(self.centered):cmul(self.buffer)          -- [x - E(x)]^2

         local buf_folded = self.buffer:view(nBatch,nFeature,iH*iW)
         --self.std:mean(self.buffer2:mean(buf_folded, 1), 3)
         self.buffer2:mean(buf_folded, 1)
         self.std:mean(self.buffer2, 3)
         self.std:add(self.eps):sqrt():pow(-1)      -- 1 / E([x - E(x)]^2)
         self.running_std:mul(1 - self.momentum):add(self.momentum, self.std) -- add to running stdv
         self.buffer:repeatTensor(self.std:view(1, nFeature, 1, 1),
                               nBatch, 1, iH, iW)

         -- divide standard-deviation + eps
         self.output:cmul(self.centered, self.buffer)
         self.normalized:copy(self.output)
      end

      if self.affine then
         -- multiply with gamma and add beta
         self.buffer:repeatTensor(self.weight:view(1, nFeature, 1, 1),
                                  nBatch, 1, iH, iW)
         self.output:cmul(self.buffer)
         self.buffer:repeatTensor(self.bias:view(1, nFeature, 1, 1),
                            nBatch, 1, iH, iW)
         self.output:add(self.buffer)
      end
   end

   return self.output
end

function BN:updateGradInput(input, gradOutput)
   assert(input:dim() == 4, 'only mini-batch supported')
   assert(gradOutput:dim() == 4, 'only mini-batch supported')
   assert(self.train == true, 'should be in training mode when self.train is true')
   local nBatch = input:size(1)
   local nFeature = input:size(2)
   local iH = input:size(3)
   local iW = input:size(4)
   local nSpatial = iH * iW


   if input:type()=='torch.CudaTensor' and validOptDimensions(nBatch, nFeature, nSpatial) then
      -- pre-allocate buffers for transpose input and output
      -- reuse same buffers from the forward pass
      self.transposedInput = self.transposedInput or input.new()
      self.transposedInput:resize(nSpatial, nFeature, nBatch)
      self.transposedOutput = self.transposedOutput or input.new()
      self.transposedOutput:resize(nSpatial, nFeature, nBatch)

      if self.affine then
         input.nn.SpatialBatchNormalization_updateGradInputAffine(self, gradOutput, nBatch, nFeature, nSpatial)
      else
         input.nn.SpatialBatchNormalization_updateGradInput(self, gradOutput, nBatch, nFeature, nSpatial)
      end
   else
      self.gradInput:cmul(self.centered, gradOutput)
      local gi_folded = self.gradInput:view(nBatch, nFeature, iH * iW)
      self.buffer2:mean(self.buffer:mean(gi_folded, 1), 3)
      self.gradInput:repeatTensor(self.buffer2:view(1, nFeature, 1, 1),
                               nBatch, 1, iH, iW)
      self.gradInput:cmul(self.centered):mul(-1)
      self.buffer:repeatTensor(self.std:view(1, nFeature, 1, 1),
                            nBatch, 1, iH, iW)

      self.gradInput:cmul(self.buffer):cmul(self.buffer)
      self.buffer:mean(gradOutput:view(nBatch, nFeature, iH*iW), 1)
      self.buffer2:mean(self.buffer, 3)
      self.buffer:repeatTensor(self.buffer2:view(1, nFeature, 1, 1),
                            nBatch, 1, iH, iW)
      self.gradInput:add(gradOutput):add(-1, self.buffer)
      self.buffer:repeatTensor(self.std:view(1, nFeature, 1, 1),
                            nBatch, 1, iH, iW)
      self.gradInput:cmul(self.buffer)

      if self.affine then
         self.buffer:repeatTensor(self.weight:view(1, nFeature, 1, 1),
                               nBatch, 1, iH, iW)
         self.gradInput:cmul(self.buffer)
      end
   end

   return self.gradInput
end

function BN:accGradParameters(input, gradOutput, scale)
   if self.affine then
      scale = scale or 1.0
      local nBatch = input:size(1)
      local nFeature = input:size(2)
      local iH = input:size(3)
      local iW = input:size(4)
      local nSpatial = iH * iW

      if input:type()=='torch.CudaTensor' and validOptDimensions(nBatch, nFeature, nSpatial) then
         input.nn.SpatialBatchNormalization_accGradParameters(self, nBatch, nFeature, nSpatial, scale)
      else
         self.buffer2:resizeAs(self.normalized):copy(self.normalized)
         self.buffer2 = self.buffer2:cmul(gradOutput):view(nBatch, nFeature, iH*iW)
         self.buffer:sum(self.buffer2, 1) -- sum over mini-batch
         self.buffer2:sum(self.buffer, 3) -- sum over pixels
         self.gradWeight:add(scale, self.buffer2)
         self.buffer:sum(gradOutput:view(nBatch, nFeature, iH*iW), 1)
         self.buffer2:sum(self.buffer, 3)
         self.gradBias:add(scale, self.buffer2) -- sum over mini-batch
      end
   end
   -- mark last phase as not forward to ensure no double calling of this kernel
   self.forwardDone = false
end
