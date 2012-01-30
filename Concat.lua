local Concat, parent = torch.class('nn.Concat', 'nn.Module')

function Concat:__init(dimension)
   parent.__init(self)
   self.modules = {}
   self.size = torch.LongStorage()
   self.dimension = dimension
end

function Concat:add(module)
   table.insert(self.modules, module)
   return self
end

function Concat:get(index)
   return self.modules[index]
end

function Concat:updateOutput(input)
   local outs = {}
   for i=1,#self.modules do
      local currentOutput = self.modules[i]:updateOutput(input)
      outs[i] = currentOutput
      if i == 1 then
         self.size:resize(currentOutput:dim()):copy(currentOutput:size())
      else
         self.size[self.dimension] = self.size[self.dimension] + currentOutput:size(self.dimension)
      end
   end
   self.output:resize(self.size)
   
   local offset = 1
   for i,module in ipairs(self.modules) do
      --local currentOutput = module:updateOutput(input)
      local currentOutput = outs[i]
      self.output:narrow(self.dimension, offset, currentOutput:size(self.dimension)):copy(currentOutput)
      offset = offset + currentOutput:size(self.dimension)
   end
   return self.output
end

function Concat:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)

   local offset = 1
   for i,module in ipairs(self.modules) do
      local currentOutput = module.output
      local currentGradInput = module:updateGradInput(input, gradOutput:narrow(self.dimension, offset, currentOutput:size(self.dimension)))
        
      if i==1 then
         self.gradInput:copy(currentGradInput)
      else
         self.gradInput:add(currentGradInput)
      end
      offset = offset + currentOutput:size(self.dimension)
   end
   return self.gradInput
end

function Concat:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   local offset = 1
   for i,module in ipairs(self.modules) do
      local currentOutput = module.output
      local currentGradInput = module:accGradParameters(input,
                                                        gradOutput:narrow(self.dimension, offset, currentOutput:size(self.dimension)),
                                                        scale)
      offset = offset + currentOutput:size(self.dimension)
   end
end

function Concat:accUpdateGradParameters(input, gradOutput, lr)
   local offset = 1
   for i,module in ipairs(self.modules) do
      local currentOutput = module.output
      local currentGradInput = module:accUpdateGradParameters(input,
                                                              gradOutput:narrow(self.dimension, offset, currentOutput:size(self.dimension)),
                                                              lr)
      offset = offset + currentOutput:size(self.dimension)
   end
end

function Concat:zeroGradParameters()
   for _,module in ipairs(self.modules) do
      module:zeroGradParameters()
   end
end

function Concat:updateParameters(learningRate)
   for _,module in ipairs(self.modules) do
      module:updateParameters(learningRate)
   end
end

function Concat:share(mlp,...)
   for i=1,#self.modules do
      self.modules[i]:share(mlp.modules[i],...); 
   end
end

function Concat:parameters()
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

function Concat:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = '  |`-> '
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = 'nn.Concat'
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
