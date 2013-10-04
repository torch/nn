local ConcatTable, parent = torch.class('nn.ConcatTable', 'nn.Module')

function ConcatTable:__init()
   parent.__init(self)
   self.modules = {}
   self.output = {}
end

function ConcatTable:add(module)
   table.insert(self.modules, module)
   return self
end

function ConcatTable:get(index)
   return self.modules[index]
end

function ConcatTable:size()
   return #self.modules 
end

function ConcatTable:updateOutput(input)
   for i=1,#self.modules do
      self.output[i] = self.modules[i]:updateOutput(input)
   end
   return self.output
end

function ConcatTable:updateGradInput(input, gradOutput)
   for i,module in ipairs(self.modules) do
      local currentGradInput = module:updateGradInput(input, gradOutput[i])
      if i == 1 then
         self.gradInput:resizeAs(currentGradInput):copy(currentGradInput)
      else
         self.gradInput:add(currentGradInput)
      end
   end
   return self.gradInput
end

function ConcatTable:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   for i,module in ipairs(self.modules) do
      module:accGradParameters(input, gradOutput[i], scale)
   end
end

function ConcatTable:accUpdateGradParameters(input, gradOutput, lr)
   for i,module in ipairs(self.modules) do
      module:accUpdateGradParameters(input, gradOutput[i], lr)
   end
end

function ConcatTable:zeroGradParameters()
   for _,module in ipairs(self.modules) do
      module:zeroGradParameters()
   end
end

function ConcatTable:updateParameters(learningRate)
   for _,module in ipairs(self.modules) do
      module:updateParameters(learningRate)
   end
end

function ConcatTable:share(mlp,...)
   for i=1,#self.modules do
      self.modules[i]:share(mlp.modules[i],...); 
   end
end

function ConcatTable:parameters()
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

function ConcatTable:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = '  |`-> '
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = 'nn.ConcatTable'
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
