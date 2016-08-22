local MapTable, parent = torch.class('nn.MapTable', 'nn.Container')

function MapTable:__init(module, shared)
   parent.__init(self)
   self.shared = shared or {'weight', 'bias', 'gradWeight', 'gradBias'}
   self.output = {}
   self.gradInput = {}
   self:add(module)
end

function MapTable:_extend(n)
   self.modules[1] = self.module
   for i = 2, n do
      if not self.modules[i] then
         self.modules[i] = self.module:clone(table.unpack(self.shared))
      end
   end
end

function MapTable:resize(n)
   self:_extend(n)
   for i = n + 1, #self.modules do
      self.modules[i] = nil
   end
end

function MapTable:add(module)
   assert(not self.module, 'Single module required')
   self.module = module
   self.modules[1] = self.module
   return self
end

function MapTable:updateOutput(input)
   self.output = {}
   self:_extend(#input)
   for i = 1, #input do
      self.output[i] = self:rethrowErrors(self.modules[i], i, 'updateOutput', input[i])
   end
   return self.output
end

function MapTable:updateGradInput(input, gradOutput)
   self.gradInput = {}
   self:_extend(#input)
   for i = 1, #input do
      self.gradInput[i] = self:rethrowErrors(self.modules[i], i, 'updateGradInput', input[i], gradOutput[i])
   end
   return self.gradInput
end

function MapTable:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   self:_extend(#input)
   for i = 1, #input do
      self:rethrowErrors(self.modules[i], i, 'accGradParameters', input[i], gradOutput[i], scale)
   end
end

function MapTable:accUpdateGradParameters(input, gradOutput, lr)
   lr = lr or 1
   self:_extend(#input)
   for i = 1, #input do
      self:rethrowErrors(self.modules[i], i, 'accUpdateGradParameters', input[i], gradOutput[i], lr)
   end
end

function MapTable:zeroGradParameters()
    if self.module then
        self.module:zeroGradParameters()
    end
end

function MapTable:updateParameters(learningRate)
    if self.module then
        self.module:updateParameters(learningRate)
    end
end

function MapTable:clearState()
   for i = 2, #self.modules do
      self.modules[i] = nil
   end
   parent.clearState(self)
end

function MapTable:__tostring__()
   local tab = '  '
   local line = '\n'
   local extlast = '      '
   local str = torch.type(self)
   if self.module then
      str = str .. ' {' .. line .. tab
      str = str .. tostring(self.module):gsub(line, line .. tab .. extlast) .. line .. '}'
   else
      str = str .. ' { }'
   end
   return str
end
