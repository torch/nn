local Parallel, parent = torch.class('nn.Parallel', 'nn.Container')

function Parallel:__init(inputDimension,outputDimension)
   parent.__init(self)
   self.modules = {}
   self.inputDimension = inputDimension
   self.outputDimension = outputDimension
end

function Parallel:updateOutput(input)
   local nModule=input:size(self.inputDimension)
   local outputs = {}
   self.totalOutputSize = self.totalOutputSize or torch.LongStorage()
   local totalOutputSize = self.totalOutputSize

   for i=1,nModule do
      local currentInput = input:select(self.inputDimension,i)
      local currentOutput = self:rethrowErrors(self.modules[i], i, 'updateOutput', currentInput)
      table.insert(outputs, currentOutput)
      local outputSize = currentOutput:size(self.outputDimension)

      if i == 1 then
         totalOutputSize:resize(currentOutput:dim()):copy(currentOutput:size())
      else
         totalOutputSize[self.outputDimension] = totalOutputSize[self.outputDimension] + outputSize
      end

   end
   self.output:resize(totalOutputSize)

   local offset = 1
   for i=1,nModule do
      local currentOutput = outputs[i]
      local outputSize = currentOutput:size(self.outputDimension)
      self.output:narrow(self.outputDimension, offset, outputSize):copy(currentOutput)
      offset = offset + currentOutput:size(self.outputDimension)
   end
   return self.output
end

function Parallel:updateGradInput(input, gradOutput)
   local nModule=input:size(self.inputDimension)
   self.gradInput:resizeAs(input)

   local offset = 1
   for i=1,nModule do
      local module=self.modules[i]
      local currentInput = input:select(self.inputDimension,i)
      local currentOutput = module.output
      local outputSize = currentOutput:size(self.outputDimension)
      local currentGradOutput = gradOutput:narrow(self.outputDimension, offset, outputSize)

      local currentGradInput = self:rethrowErrors(module, i, 'updateGradInput', currentInput, currentGradOutput)

      self.gradInput:select(self.inputDimension,i):copy(currentGradInput)
      offset = offset + outputSize
   end
   return self.gradInput
end

function Parallel:accGradParameters(input, gradOutput, scale)
   local nModule=input:size(self.inputDimension)

   local offset = 1
   for i=1,nModule do
      local module = self.modules[i]
      local currentOutput = module.output
      local outputSize = currentOutput:size(self.outputDimension)

      self:rethrowErrors(module, i, 'accGradParameters',
          input:select(self.inputDimension,i),
          gradOutput:narrow(self.outputDimension, offset,outputSize),
          scale)

      offset = offset + outputSize
   end
end

function Parallel:accUpdateGradParameters(input, gradOutput, lr)
   local nModule=input:size(self.inputDimension)

   local offset = 1
   for i=1,nModule do
      local module = self.modules[i];
      local currentOutput = module.output
      self:rethrowErrors(module, i, 'accUpdateGradParameters',
          input:select(self.inputDimension,i),
          gradOutput:narrow(self.outputDimension, offset,
                            currentOutput:size(self.outputDimension)),
          lr)

      offset = offset + currentOutput:size(self.outputDimension)
   end
end

function Parallel:__tostring__()
   local g = function(s) -- GREEN
      if nn.config.prettyPrint then return '\27[0;32m' .. s .. '\27[0m' end
      return s
   end
   local tab = '  '
   local line = '\n'
   local next = g '  |`-> '
   local ext = g '  |    '
   local extlast = '       '
   local last = g '   ... -> '
   local str = g(torch.type(self))
   str = str .. g ' {' .. line .. tab .. g 'input'
   for i=1,#self.modules do
      if i == #self.modules then
         str = str .. line .. tab .. next .. g '(' .. i .. g '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. extlast)
      else
         str = str .. line .. tab .. next .. g '(' .. i .. g '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. ext)
      end
   end
   str = str .. line .. tab .. last .. g 'output'
   str = str .. line .. g '}'
   return str
end
