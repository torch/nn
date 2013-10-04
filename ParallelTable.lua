local ParallelTable, parent = torch.class('nn.ParallelTable', 'nn.Module')

function ParallelTable:__init()
   parent.__init(self)
   self.modules = {}
   self.output = {}
   self.gradInput = {}
end

function ParallelTable:add(module)
   table.insert(self.modules, module)
   return self
end

function ParallelTable:get(index)
   return self.modules[index]
end

function ParallelTable:size()
   return #self.modules 
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

function ParallelTable:zeroGradParameters()
   for _,module in ipairs(self.modules) do
      module:zeroGradParameters()
   end
end

function ParallelTable:updateParameters(learningRate)
   for _,module in ipairs(self.modules) do
      module:updateParameters(learningRate)
   end
end

function ParallelTable:share(mlp,...)
   for i=1,#self.modules do
      self.modules[i]:share(mlp.modules[i],...); 
   end
end

function ParallelTable:parameters()
   local function tinsert(to, from)
      if type(from) == 'table' then
         for i=1,#from do
            tinsert(to,from[i])
         end
      else
         table.insert(to,from)
      end
   end
   local w = {}
   local gw = {}
   for i=1,#self.modules do
      local mw,mgw = self.modules[i]:parameters()
      if mw then
         tinsert(w,mw)
         tinsert(gw,mgw)
      end
   end
   return w,gw
end

function ParallelTable:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = '  |`-> '
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = 'nn.ParallelTable'
   str = str .. ' {' .. line .. tab .. 'input'
   for i=1,#self.modules do
      if i == self.modules then
         str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. extlast)
      else
         str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. ext)
      end
   end
   str = str .. line .. tab .. last .. 'output'
   str = str .. line .. '}'
   return str
end
