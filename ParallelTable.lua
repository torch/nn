local ParallelTable, parent = torch.class('nn.ParallelTable', 'nn.Container')

function ParallelTable:__init()
   parent.__init(self)
   self.modules = {}
   self.output = {}
   self.gradInput = {}
end

function ParallelTable:updateOutput(input)
   for i=1,#self.modules do
      self.output[i] = self.modules[i]:updateOutput(input[i])
   end
   return self.output
end

function ParallelTable:updateGradInput(input, gradOutput)
   for i,module in ipairs(self.modules) do
      self.gradInput[i]= module:updateGradInput(input[i], gradOutput[i])
   end
   return self.gradInput
end

function ParallelTable:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   for i,module in ipairs(self.modules) do
      module:accGradParameters(input[i], gradOutput[i], scale)
   end
end

function ParallelTable:accUpdateGradParameters(input, gradOutput, lr)
   lr = lr or 1
   for i,module in ipairs(self.modules) do
      module:accUpdateGradParameters(input[i], gradOutput[i], lr)
   end
end

function ParallelTable:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = '  |`-> '
   local ext = '  |    '
   local last = '   ... -> '
   local str = torch.type(self)
   str = str .. ' {' .. line .. tab .. 'input'
   for i=1,#self.modules do
      str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. ext)
   end
   str = str .. line .. tab .. last .. 'output'
   str = str .. line .. '}'
   return str
end
