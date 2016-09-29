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
   self.keys = torch.LongTensor()
   self.updateKeysBuffer = torch.LongTensor()
   self.values = torch.Tensor()
   self.sizes = torch.LongTensor()
   self.cumSumSizes = torch.LongTensor()
   self.runningKeys = {}
   self.runningGradWeights = {}

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

   return lkeys, lvalues
end

function IndexLinear:updateOutput(input)
   self.lkeys, self.lvalues = self:reshapeInput(input)
   self.runningKeys[self.runningCter] = self.runningKeys[self.runningCter] or torch.LongTensor()
   self.keys = self.runningKeys[self.runningCter]
   self.keys = self.keys:type('torch.LongTensor')
   local marker = 0

   self.sizes = self.sizes:type('torch.LongTensor')
   self.cumSumSizes = self.cumSumSizes:type('torch.LongTensor')
   self.sizes:resize(#self.lkeys)
   self.cumSumSizes:resize(#self.lkeys)
   self.cumSumSizes[1] = 0
   for i=1,#self.lkeys do
      assert(self.lvalues[i]:dim() == 1 and self.lkeys[i]:dim() == 1, "keys and values should be 1D")
      self.sizes[i] = self.lkeys[i]:size(1)
      if i > 1 then
         self.cumSumSizes[i] = self.cumSumSizes[i-1] + self.sizes[i-1]
      end
   end
   self.keys:cat(self.lkeys, 1)
   self.values:cat(self.lvalues, 1)

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
      gradOutput:cdata(),
      self.weight:cdata(),
      self.bias:cdata(),
      self.weightDecay or 0,
      scale or 1
   )
end

function IndexLinear:accGradParameters(input, gradOutput, scale)
   -- Same as the runningKeys in the updateOutput function,
   -- get a table of dense runningGradWeights
   self.runningGradWeights[self.runningCter] = self.runningGradWeights[self.runningCter] or self.values.new()
   self.values.THNN.IndexLinear_accGradParameters(
      self.keys:cdata(),
      self.offset,
      self.normalize > 0 and self.normalizedValues:cdata() or self.values:cdata(),
      self.sizes:cdata(),
      gradOutput:cdata(),
      self.runningGradWeights[self.runningCter]:cdata(),
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
   self.gradInput = {{},{}}
   -- Revamped from nn.SparseLinear.updateGradInput
   if self.doGradInput and self.normalize > 0 then
      error('updateGradInput is not implemented in max-normalize mode')
   end
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
      local ini = self.weight:size(1)

      local indices = torch.range(1, ini)
      for i = 1,gi:size(1) do
         self.gradInput[1][i] = torch.LongTensor(ini)
         self.gradInput[1][i]:copy(indices)
         self.gradInput[2][i] = gradOutput.new(ini)
         self.gradInput[2][i]:copy(gi[i])
      end
   end
   if self.noBatch then
      self.gradInput = {self.gradInput[1][1], self.gradInput[2][1]}
   end
   return self.gradInput
end

function IndexLinear:updateParameters(lr)
   if self.runningCter > 1 then
      if self.runningCter == 2 then
         self.updateKeys = self.runningKeys[1]
         self.gradWeight = self.runningGradWeights[1]
      else
         local lkeys = {}
         local lgweights = {}
         local totalSize = 0
         for i=1,self.runningCter-1 do
            lkeys[i] = self.runningKeys[i]
            -- Change layout to take advantage of the 1-D contiguous torch.cat
            lgweights[i] = self.runningGradWeights[i]:contiguous()
            lgweights[i]:resize(lgweights[i]:nElement())
            totalSize = totalSize + lkeys[i]:size(1)
         end
         self.updateKeysBuffer:cat(lkeys, 1)
         self.gradWeightBuffer:cat(lgweights, 1)
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
   self.keys = torch.LongTensor()
   self.zerokeys = nil
   self.updateKeys = nil
   self.values = torch.Tensor()
   self.sizes = torch.LongTensor()
   self.lkeys = {}
   self.lvalues = {}
   self.gradWeightBuffer = torch.Tensor()
   self.valuesBuffer = torch.Tensor()
   self.updateKeysBuffer = torch.LongTensor()
   self.values = torch.Tensor()
   return parent.clearState(self)
end
