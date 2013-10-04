local Parallel, parent = torch.class('nn.Parallel', 'nn.Module')

function Parallel:__init(inputDimension,outputDimension)
   parent.__init(self)
   self.modules = {}
   self.size = torch.LongStorage() 
   self.inputDimension = inputDimension
   self.outputDimension = outputDimension
end

function Parallel:add(module)
   table.insert(self.modules, module)
   return self
end

function Parallel:get(index)
   return self.modules[index]
end

function Parallel:updateOutput(input)
   
   local modules=input:size(self.inputDimension)

   for i=1,modules do
      local currentOutput = 
	self.modules[i]:updateOutput(input:select(self.inputDimension,i))
      
      if i == 1 then
         self.size:resize(currentOutput:dim()):copy(currentOutput:size())
      else
         self.size[self.outputDimension] = self.size[self.outputDimension] 
				     + currentOutput:size(self.outputDimension)
      end
   end
   self.output:resize(self.size)
   
   local offset = 1
   for i=1,modules do
      local currentOutput = self.modules[i]:updateOutput(input:select(self.inputDimension,i))

      self.output:narrow(self.outputDimension, offset, 
	                 currentOutput:size(self.outputDimension)):copy(currentOutput)
      offset = offset + currentOutput:size(self.outputDimension)
   end 
   return self.output
end

function Parallel:updateGradInput(input, gradOutput)
   local nModule=input:size(self.inputDimension)
   self.gradInput:resizeAs(input)

   local offset = 1
   for i=1,nModule do 
      local module=self.modules[i];
      local currentOutput = module.output
      local currentGradInput = 
	module:updateGradInput(input:select(self.inputDimension,i),
                        gradOutput:narrow(self.outputDimension, 
                                          offset, currentOutput:size(self.outputDimension)))
        
      self.gradInput:select(self.inputDimension,i):copy(currentGradInput)
      offset = offset + currentOutput:size(self.outputDimension)
   end
   return self.gradInput
end

function Parallel:accGradParameters(input, gradOutput, scale)
   local nModule=input:size(self.inputDimension)

   local offset = 1
   for i=1,nModule do 
      local module = self.modules[i];
      local currentOutput = module.output
      local currentGradInput = 
         module:accGradParameters(input:select(self.inputDimension,i),
                                  gradOutput:narrow(self.outputDimension, 
                                                    offset, currentOutput:size(self.outputDimension)), scale)
        
      offset = offset + currentOutput:size(self.outputDimension)
   end
end

function Parallel:accUpdateGradParameters(input, gradOutput, lr)
   local nModule=input:size(self.inputDimension)

   local offset = 1
   for i=1,nModule do 
      local module = self.modules[i];
      local currentOutput = module.output
      local currentGradInput = 
         module:accUpdateGradParameters(input:select(self.inputDimension,i),
                                        gradOutput:narrow(self.outputDimension, 
                                                          offset, currentOutput:size(self.outputDimension)), lr)
        
      offset = offset + currentOutput:size(self.outputDimension)
   end
end
 
function Parallel:zeroGradParameters()
   for _,module in ipairs(self.modules) do
      module:zeroGradParameters()
   end
end

function Parallel:updateParameters(learningRate)
   for _,module in ipairs(self.modules) do
      module:updateParameters(learningRate)
   end
end

function Parallel:share(mlp,...)
   for i=1,#self.modules do
      self.modules[i]:share(mlp.modules[i],...); 
   end
end

function Parallel:parameters()
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

function Parallel:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = '  |`-> '
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = 'nn.Parallel'
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
