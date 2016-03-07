local Sequential, _ = torch.class('nn.Sequential', 'nn.Container')

function Sequential:__len()
   return #self.modules
end

function Sequential:add(module)
   if #self.modules == 0 then
      self.gradInput = module.gradInput
   end
   table.insert(self.modules, module)
   self.output = module.output
   return self
end

function Sequential:insert(module, index)
   index = index or (#self.modules + 1)
   if index > (#self.modules + 1) or index < 1 then
      error"index should be contiguous to existing modules"
   end
   table.insert(self.modules, index, module)
   self.output = self.modules[#self.modules].output
   self.gradInput = self.modules[1].gradInput
end

function Sequential:remove(index)
   index = index or #self.modules
   if index > #self.modules or index < 1 then
      error"index out of range"
   end
   table.remove(self.modules, index)
   if #self.modules > 0 then
       self.output = self.modules[#self.modules].output
       self.gradInput = self.modules[1].gradInput
   else
       self.output = torch.Tensor()
       self.gradInput = torch.Tensor()
   end
end

function Sequential:updateOutput(input)
   local currentOutput = input
   for i=1,#self.modules do
      currentOutput = self.modules[i]:updateOutput(currentOutput)
   end
   self.output = currentOutput
   return currentOutput
end

function Sequential:updateGradInput(input, gradOutput)
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentGradOutput = currentModule:updateGradInput(previousModule.output, currentGradOutput)
      currentModule = previousModule
   end
   currentGradOutput = currentModule:updateGradInput(input, currentGradOutput)
   self.gradInput = currentGradOutput
   return currentGradOutput
end

function Sequential:accGradParameters(input, gradOutput, scale)
   scale = scale or 1

   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentModule:accGradParameters(previousModule.output, currentGradOutput, scale)
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
   end

   currentModule:accGradParameters(input, currentGradOutput, scale)
end

function Sequential:backward(input, gradOutput, scale)
   scale = scale or 1
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentGradOutput = currentModule:backward(previousModule.output, currentGradOutput, scale)
      currentModule.gradInput = currentGradOutput
      currentModule = previousModule
   end
   currentGradOutput = currentModule:backward(input, currentGradOutput, scale)
   self.gradInput = currentGradOutput
   return currentGradOutput
end

function Sequential:accUpdateGradParameters(input, gradOutput, lr)
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentModule:accUpdateGradParameters(previousModule.output, currentGradOutput, lr)
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
   end

   currentModule:accUpdateGradParameters(input, currentGradOutput, lr)
end


function Sequential:__tostring__()
   local b = function(s) return s end
   if nn.config.prettyPrint then b = require('trepl.colorize').blue end
   local tab = '  '
   local line = '\n'
   local next = b ' -> '
   local str = b 'nn.Sequential'
   str = str .. b ' {' .. line .. tab .. b '[input'
   for i=1,#self.modules do
      str = str .. next .. b '(' .. i .. b ')'
   end
   str = str .. next .. b 'output]'
   for i=1,#self.modules do
      str = str .. line .. tab .. b '(' .. i .. b '): ' .. tostring(self.modules[i]):gsub(line, line .. tab)
   end
   str = str .. line .. b '}'
   return str
end
