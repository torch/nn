local ffi  = require 'ffi'
local IndexLinear, parent = torch.class('nn.IndexLinear', 'nn.Module')

function IndexLinear:__init(inputSize, outputSize, doGradInput, keysOffset, weight, bias, normalize)
   parent.__init(self)

   self.normalize = normalize and 4 or 0

   -- This is important to keep the possibility of sharing a weight
   -- directly, without having to allocate it first.
   -- The reason is these weights can be very large.
   self.weight = weight or torch.Tensor(inputSize, outputSize + self.normalize):zero()
   self.bias = bias or torch.Tensor(outputSize):zero()
   self.inputSize = self.weight and self.weight:size(1) or inputSize
   self.outputSize = self.weight and (self.weight:size(2)-self.normalize) or outputSize

   -- gradWeight is not initialized as we're doing dense gradient accumulation
   -- This is more efficient and avoids allocating a giant useless gradWeight
   self.gradWeight = torch.Tensor()

   -- gradBias still works the same as it's already dense
   self.gradBias = torch.Tensor(self.outputSize):zero()
   self.gradBiasBuffer = torch.Tensor()

   -- Buffers
   self.gradWeightBuffer = torch.Tensor()
   self.valuesBuffer = torch.Tensor()
   self.normalizedValues = torch.Tensor()
   self.runningCumSumSizes = {}
   self.runningKeys = {}
   self.runningGradWeights = {}

   -- self.sizes, self.cumSumSizes are calculated on the CPU even when using CUDA.
   -- These two tables make it easier to resize these buffers instead of re-allocating them.
   -- self.*Cache[1] always contains values on CPU.
   -- If CUDA is being used, self.*Cache[2] contains values on GPU.
   self.sizesCache = {}
   self.cumSumSizesCache = {}

   -- A few options
   self.weightDecay = 0
   self.doGradInput = doGradInput or false
   self.offset = keysOffset and keysOffset-1 or -1 -- if this adds self.offset to indices

   -- That is used to accumulate keys and gradWeights
   -- when doing gradients accumulations
   self.runningCter = 1
end

function IndexLinear:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv):mul(0.000001)
   if self.normalize and self.normalize > 0 then
      self.weight[{{}, {1,self.normalize}}]:zero()
   end
end

function IndexLinear:reshapeInput(input)
   assert(type(input) == 'table')

   local ninputs = 0
   for _, v in ipairs(input) do
      ninputs = ninputs + 1
   end

   assert(ninputs == 2 or ninputs == 3)

   if ninputs == 3 then
      local fkeys = input[1]
      local fvals = input[2]
      local fsizes = torch.isTensor(input[3]) and input[3] or fkeys.new{input[3]}
      assert(fkeys:nElement() == fvals:nElement(), 'Keys and values should be of same size')
      assert(fkeys:dim() == 1, 'Keys and values should be 1D')
      self.isFlat = true
      self.noBatch = false
      return fkeys, fvals, fsizes
   end

   local keys = input[1]
   local values = input[2]
   local lkeys, lvalues
   if type(keys) == 'table' and type(values) == 'table' then
      lkeys, lvalues = keys, values
      self.noBatch = false
   elseif torch.isTensor(keys) and torch.isTensor(values) then
      lkeys, lvalues = {keys}, {values}
      self.noBatch = true
   else
      error('Wrong input format.')
   end

   for i=1,#lkeys do
      assert(lvalues[i]:dim() == 1 and lkeys[i]:dim() == 1, "keys and values should be 1D")
   end

   return lkeys, lvalues
end

function IndexLinear:longTensor(...)
   if (self:type() == 'torch.CudaTensor') then
      return torch.CudaLongTensor(...)
   else
      return torch.LongTensor(...)
   end
end

function IndexLinear:flattenInputs(input)
   local lkeys, lvalues, sizes = self:reshapeInput(input)

   local counter = self.runningCter

   -- Ensure everything is of the right type
   local isCuda = (self:type() == 'torch.CudaTensor')
   self.runningKeys[counter] = self.runningKeys[counter] or self:longTensor()
   self.keys = self.runningKeys[counter]

   if self.isFlat then
      self.values = self.values or lvalues.new()
      self.sizes = self.sizes or self:longTensor()

      self.keys:resize(lkeys:size()):copy(lkeys)
      self.values:resize(lvalues:size()):copy(lvalues)
      self.sizes = sizes
      self.cumSumSizes = self.cumSumSizes or self.sizes.new()
      self.cumSumSizes:cumsum(self.sizes)
   else
      self.values = self.values or lvalues[1].new()

      self.lkeys = lkeys
      self.lvalues = lvalues
      local batchSize = #self.lkeys

      self.sizesCache[1] = self.sizesCache[1] or torch.LongTensor(batchSize)
      self.cumSumSizesCache[1] = self.cumSumSizesCache[1] or torch.LongTensor(batchSize)

      self.sizes = self.sizesCache[1]
      self.cumSumSizes = self.cumSumSizesCache[1]

      self.sizes:resize(batchSize)
      self.cumSumSizes:resize(batchSize)

      for i = 1,batchSize do
         self.sizes[i] = self.lkeys[i]:size(1)
      end
      self.cumSumSizes:cumsum(self.sizes)

      self.keys:cat(self.lkeys, 1)
      self.values:cat(self.lvalues, 1)

      if isCuda then
         -- Get the GPU cache
         self.sizesCache[2] = self.sizesCache[2] or torch.CudaLongTensor()
         self.cumSumSizesCache[2] = self.cumSumSizesCache[2] or torch.CudaLongTensor()

         self.sizes = self.sizesCache[2]
         self.cumSumSizes = self.cumSumSizesCache[2]

         -- Resize and copy to GPU
         self.sizes:resize(batchSize):copy(self.sizesCache[1])
         self.cumSumSizes:resize(batchSize):copy(self.cumSumSizesCache[1])
      end
   end
   self.runningCumSumSizes[counter] = self.cumSumSizes
end

function IndexLinear:updateOutput(input)

   self:flattenInputs(input)

   self.values.THNN.IndexLinear_updateOutput(
      self.keys:cdata(),
      self.offset,
      self.values:cdata(),
      self.sizes:cdata(),
      self.cumSumSizes:cdata(),
      self.output:cdata(),
      self.weight:cdata(),
      self.bias:cdata(),
      self.normalizedValues:cdata(),
      self.train and 1 or 0
      )

   if self.noBatch then
      self.output:resize(self.output:size(2))
   end
   return self.output
end

function IndexLinear:accUpdateGradParameters(input, gradOutput, scale)
   self.values.THNN.IndexLinear_accUpdateGradParameters(
      self.keys:cdata(),
      self.offset,
      self.normalize > 0 and self.normalizedValues:cdata() or self.values:cdata(),
      self.sizes:cdata(),
      self.cumSumSizes:cdata(),
      gradOutput:cdata(),
      self.weight:cdata(),
      self.bias:cdata(),
      self.weightDecay or 0,
      scale or 1
   )
end

function IndexLinear:accGradParameters(input, gradOutput, scale)

   local counter = self.runningCter

   -- Same as the runningKeys in the updateOutput function,
   -- get a table of dense runningGradWeights
   self.runningGradWeights[counter] = self.runningGradWeights[counter] or self.values.new()
   self.values.THNN.IndexLinear_accGradParameters(
      self.keys:cdata(),
      self.offset,
      self.normalize > 0 and self.normalizedValues:cdata() or self.values:cdata(),
      self.sizes:cdata(),
      self.cumSumSizes:cdata(),
      gradOutput:cdata(),
      self.runningGradWeights[counter]:cdata(),
      self.gradBias:cdata(),
      self.weight:cdata(),
      self.bias:cdata(),
      self.valuesBuffer:cdata(),
      self.weightDecay or 0,
      scale or 1
   )

   -- Increment the running counter to create a new buffer
   -- if we don't flush them in zerogradParamters
   self.runningCter = self.runningCter + 1
end

function IndexLinear:updateGradInput(input, gradOutput)
   self.gradInput = {}
   -- Revamped from nn.SparseLinear.updateGradInput
   if self.doGradInput and self.normalize > 0 then
      error('updateGradInput is not implemented in max-normalize mode')
   end

   local ini = self.weight:size(1)

   if self.doGradInput then
      local gi = gradOutput.new()
      if gradOutput:dim() == 1 then
         gi:resize(self.weight:size(1))
         gi:mv(self.weight,gradOutput)
         gi:resize(1, self.weight:size(1))
      elseif gradOutput:dim() == 2 then
         gi:resize(gradOutput:size(1), self.weight:size(1))
         gi:mm(gradOutput, self.weight:t())
      end

      local indices = self.runningKeys[1].new(ini):range(1, ini)

      if self.isFlat then
         self.gradInput[1] = torch.repeatTensor(indices, gi:size(1), 1)
         self.gradInput[2] = gi
      else
         self.gradInput[1] = {}
         self.gradInput[2] = {}
         for i = 1,gi:size(1) do
            self.gradInput[1][i] = self.runningKeys[1].new(ini)
            self.gradInput[1][i]:copy(indices)
            self.gradInput[2][i] = gradOutput.new(ini)
            self.gradInput[2][i]:copy(gi[i])
         end
      end
   end

   if self.noBatch then
      if self.isFlat then
         self.gradInput = {self.gradInput[1]:resize(ini), self.gradInput[2]:resize(ini)}
      else
         self.gradInput = {self.gradInput[1][1], self.gradInput[2][1]}
      end
   end
   return self.gradInput
end

function IndexLinear:updateParameters(lr)
   local counter = self.runningCter
   if counter > 1 then
      if counter == 2 then
         self.updateKeys = self.runningKeys[1]
         self.gradWeight = self.runningGradWeights[1]
      else
         self.updateKeysBuffer = self.updateKeysBuffer or self:longTensor()
         local lkeys = {}
         local lgweights = {}
         local totalSize = 0
         local lCumSumSizes = {}
         for i=1,counter-1 do
            lkeys[i] = self.runningKeys[i]
            -- Change layout to take advantage of the 1-D contiguous torch.cat
            lgweights[i] = self.runningGradWeights[i]:contiguous()
            lgweights[i]:resize(lgweights[i]:nElement())
            totalSize = totalSize + lkeys[i]:size(1)
            lCumSumSizes[i] = self.runningCumSumSizes[i]
            -- runningCumSumSizes[i] is an array of size batchSize + 1
            -- The last element contains the value lkeys[i]:size(1)
            -- We need to remove this last element for all entries except the last one
            local batchSize = lCumSumSizes[i]:nElement()
            if (i < counter - 1) then
               lCumSumSizes[i] = lCumSumSizes[i][{{1, batchSize}}]
            end
            -- The running total needs to be added to the current entry
            lCumSumSizes[i] = (i > 1 and totalSize or 0) + lCumSumSizes[i]
         end
         self.updateKeysBuffer:cat(lkeys, 1)
         self.gradWeightBuffer:cat(lgweights, 1)
         self.cumSumSizes:cat(lCumSumSizes, 1)
         self.gradWeightBuffer:resize(totalSize, self.outputSize)
         self.gradWeight = self.gradWeightBuffer
         self.updateKeys = self.updateKeysBuffer
      end
      self.values.THNN.IndexLinear_updateParameters(
            self.gradWeight:cdata(),
            self.gradBias:cdata(),
            self.weight:cdata(),
            self.bias:cdata(),
            self.updateKeys:cdata(),
            self.cumSumSizes:cdata(),
            self.offset,
            self.weightDecay or 0,
            lr or error('You must specify a learning rate')
         )
   end
end

function IndexLinear:zeroGradParameters()
   -- No need to do anything here as gradWeight is dense
   self.gradBias:zero()

   -- The below piece of code would reset
   -- the smart scaling parameters for each features
   -- each time we call zeroGradParameters
   -- TODO: decide what to do with that piece of code.
   -- NB: this should be commented along with the corresponding
   -- piece of code in lib/THNN/generic/IndexLinear.c, in the accUpdateGradParameters function.

   --[[
   local w = self.weight:select(2, 3)
   if self.updateKeys and self.updateKeys:nElement() > 0 then
      self.updateKeysBuffer:resizeAs(self.updateKeys):copy(self.updateKeys):add(self.offset+1)
      w:indexFill(1, self.updateKeysBuffer, 0)
   end
   ]]--
   self.runningCter = 1
end

function IndexLinear:clearState()
   self.runningKeys = {}
   self.runningGradWeights = {}
   self.keys = nil
   self.zerokeys = nil
   self.updateKeys = nil
   self.values = nil
   self.sizes = nil
   self.lkeys = {}
   self.lvalues = {}
   self.gradWeightBuffer = self.gradWeightBuffer.new()
   self.valuesBuffer = self.valuesBuffer.new()
   self.updateKeysBuffer = nil
   self.values = nil
   return parent.clearState(self)
end
