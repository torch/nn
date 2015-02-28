------------------------------------------------------------------------
--[[ Recurrent ]]--
-- Ref. A.: http://goo.gl/vtVGkO (Mikolov et al.)
-- B. http://goo.gl/hu1Lqm
-- Processes the sequence one timestep (forward/backward) at a time. 
-- A call to backward only keeps a log of the gradOutputs and scales.
-- Back-Propagation Through Time (BPTT) is done when updateParameters
-- is called. The Module keeps a list of all previous representations 
-- (Module.outputs), including intermediate ones for BPTT.
-- To use this module with batches, we suggest using different 
-- sequences of the same size within a batch and calling 
-- updateParameters() at the end of the Sequence. 
-- Note that this container won't work with modules that use more than the
-- output attribute to keep track of their internal state between 
-- forward and backward. For example, you cannot nest Recurrent Modules.
-- (This could be alleviated by added Module:getState/setState methods.)
------------------------------------------------------------------------
local Recurrent, parent = torch.class('nn.Recurrent', 'nn.Container')

function Recurrent:__init(start, input, feedback, transfer, rho, merge)
   parent.__init(self)
   
   local ts = torch.type(start)
   if ts == 'torch.LongTensor' or ts == 'number' then
      start = nn.Add(start)
   end
   
   self.startModule = start
   self.inputModule = input
   self.feedbackModule = feedback
   self.transferModule = transfer or nn.Sigmoid()
   self.mergeModule = merge or nn.CAddTable()
   self.rho = rho or 5
   
   -- used for the first step 
   self.initialModule = nn.Sequential()
   self.initialModule:add(self.inputModule)
   self.initialModule:add(self.startModule)
   self.initialModule:add(self.transferModule)
   
   -- used for the other steps (steps > 1)
   local parallelModule = nn.ParallelTable()
   parallelModule:add(self.inputModule)
   parallelModule:add(self.feedbackModule)
   self.recurrentModule = nn.Sequential()
   self.recurrentModule:add(parallelModule)
   self.recurrentModule:add(self.mergeModule)
   self.recurrentModule:add(self.transferModule)
   
   self.modules = {self.startModule, self.inputModule, self.feedbackModule, self.transferModule, self.mergeModule}
   
   self.initialOutputs = {}
   self.initialGradInputs = {}
   self.recurrentOutputs = {}
   self.recurrentGradInputs = {}
   
   self.fastBackward = true
   self.copyInputs = true
   
   self.inputs = {}
   self.outputs = {}
   self.gradOutputs = {}
   self.scales = {}
   
   self.gradParametersAccumulated = false
   self.step = 1
   
   self:reset()
end

local function recursiveClone(t)
   local clone
   if torch.type(t) == 'table' then
      clone = {}
      for i = 1, #t do
         clone[i] = recursiveClone(t[i])
      end
   else
      if torch.typename(t) and 
        torch.typename(t):find('torch%..+Tensor') then
         clone = t:clone()
      end
   end
   return clone
end

function Recurrent:updateOutput(input)
   -- output(t) = transfer(feedback(output_(t-1)) + input(input_(t)))
   local output
   if self.step == 1 then
      -- set/save the output states
      local modules = self.initialModule:listModules()
      for i,modula in ipairs(modules) do
         local output_ = self.initialOutputs[i] or recursiveClone(modula.output)
         modula.output = output_
      end
      output = self.initialModule:updateOutput(input)
      for i,modula in ipairs(modules) do
         self.initialOutputs[i]  = modula.output
      end
   else
      if self.train ~= false then
         -- set/save the output states
         local modules = self.recurrentModule:listModules()
         self:recycle()
         local recurrentOutputs = self.recurrentOutputs[self.step]
         if not recurrentOutputs then
            recurrentOutputs = {}
            self.recurrentOutputs[self.step] = recurrentOutputs
         end
         for i,modula in ipairs(modules) do
            local output_ = recurrentOutputs[i] or recursiveClone(modula.output)
            modula.output = output_
         end
          -- self.output is the previous output of this module
         output = self.recurrentModule:updateOutput{input, self.output}
         for i,modula in ipairs(modules) do
            recurrentOutputs[i]  = modula.output
         end
      else
         -- self.output is the previous output of this module
         output = self.recurrentModule:updateOutput{input, self.output}
      end
   end
   
   if self.train ~= false then
      local input_ = self.inputs[self.step]
      if not input_ then
         input_ = input.new()
         self.inputs[self.step] = input_
      end
      if self.copyInputs then
         input_:resizeAs(input):copy(input)
      else
         input_:set(input)
      end
   end
   
   self.outputs[self.step] = output
   self.output:set(output)
   self.step = self.step + 1
   self.gradParametersAccumulated = false
   return self.output
end

function Recurrent:updateGradInput(input, gradOutput)
   -- Back-Propagate Through Time (BPTT) happens in updateParameters()
   -- for now we just keep a list of the gradOutputs
   local gradOutput_ = self.gradOutputs[self.step-1] 
   if not gradOutput_ then
      gradOutput_ = recursiveClone(gradOutput)
      self.gradOutputs[self.step-1] = gradOutput_
   end
   gradOutput_:resizeAs(gradOutput):copy(gradOutput)
end

function Recurrent:accGradParameters(input, gradOutput, scale)
   -- Back-Propagate Through Time (BPTT) happens in updateParameters()
   -- for now we just keep a list of the scales
   self.scales[self.step-1] = scale
end

-- not to be confused with the hit movie Back to the Future
function Recurrent:backwardThroughTime()
   assert(self.step > 1, "expecting at least one updateOutput")
   local rho = math.min(self.rho, self.step-1)
   local stop = self.step - rho
   if self.fastBackward then
      local gradInput
      for step=self.step-1,math.max(stop, 2),-1 do
         -- set the output/gradOutput states of current Module
         local modules = self.recurrentModule:listModules()
         local recurrentOutputs = self.recurrentOutputs[step]
         local recurrentGradInputs = self.recurrentGradInputs[step]
         if not recurrentGradInputs then
            recurrentGradInputs = {}
            self.recurrentGradInputs[step] = recurrentGradInputs
         end
         for i,modula in ipairs(modules) do
            local output, gradInput = modula.output, modula.gradInput
            local output_ = recurrentOutputs[i]
            local gradInput_ = recurrentGradInputs[i] or recursiveClone(gradInput)
            assert(output_, "backwardThroughTime should be preceded by updateOutput")
            assert(gradInput_, "missing gradInput")
            modula.output = output_
            modula.gradInput = gradInput_
         end
         
         -- backward propagate through this step
         local input = self.inputs[step]
         local output = self.outputs[step-1]
         local gradOutput = self.gradOutputs[step]
         if gradInput then
            gradOutput:add(gradInput)
         end
         local scale = self.scales[step]
         
         gradInput = self.recurrentModule:backward({input, output}, gradOutput, scale/rho)[2]
         for i,modula in ipairs(modules) do
            recurrentGradInputs[i] = modula.gradInput
         end
      end
      
      if stop <= 1 then
         -- set the output/gradOutput states of initialModule
         local modules = self.initialModule:listModules()
         for i,modula in ipairs(modules) do
            local output, gradInput = modula.output, modula.gradInput
            local output_ = self.initialOutputs[i]
            local gradInput_ = self.initialGradInputs[i] or recursiveClone(gradInput)
            modula.output = output_
            modula.gradInput = gradInput_
         end
         
         -- backward propagate through first step
         local input = self.inputs[1]
         local gradOutput = self.gradOutputs[1]
         if gradInput then
            gradOutput:add(gradInput)
         end
         local scale = self.scales[1]
         gradInput = self.initialModule:backward(input, gradOutput, scale/rho)
         
         for i,modula in ipairs(modules) do
            self.initialGradInputs[i] = modula.gradInput
         end
         
         -- startModule's gradParams shouldn't be step-averaged
         -- as it is used only once. So un-step-average it
         local params, gradParams = self.startModule:parameters()
         if gradParams then
            for i,gradParam in ipairs(gradParams) do
               gradParam:mul(rho)
            end
         end
         
         self.gradParametersAccumulated = true
         return gradInput
      end
   else
      local gradInput = self:updateGradInputThroughTime()
      self:accGradParametersThroughTime()
      return gradInput
   end
end

function Recurrent:backwardUpdateThroughTime(learningRate)
   local gradInput = self:updateGradInputThroughTime()
   self:accUpdateGradParametersThroughTime(learningRate)
   return gradInput
end

function Recurrent:updateGradInputThroughTime()
   assert(self.step > 1, "expecting at least one updateOutput")
   local gradInput
   local rho = math.min(self.rho, self.step-1)
   local stop = self.step - rho
   for step=self.step-1,math.max(stop,2),-1 do
      -- set the output/gradOutput states of current Module
      local modules = self.recurrentModule:listModules()
      local recurrentOutputs = self.recurrentOutputs[step]
      local recurrentGradInputs = self.recurrentGradInputs[step]
      if not recurrentGradInputs then
         recurrentGradInputs = {}
         self.recurrentGradInputs[step] = recurrentGradInputs
      end
      for i,modula in ipairs(modules) do
         local output, gradInput = modula.output, modula.gradInput
         local output_ = recurrentOutputs[i]
         local gradInput_ = recurrentGradInputs[i] or recursiveClone(gradInput)
         assert(output_, "updateGradInputThroughTime should be preceded by updateOutput")
         modula.output = output_
         modula.gradInput = gradInput_
      end
      
      -- backward propagate through this step
      local input = self.inputs[step]
      local output = self.outputs[step-1]
      local gradOutput = self.gradOutputs[step]
      if gradInput then
         gradOutput:add(gradInput)
      end
      gradInput = self.recurrentModule:updateGradInput({input, output}, gradOutput)[2]
      for i,modula in ipairs(modules) do
         recurrentGradInputs[i] = modula.gradInput
      end
   end
   
   if stop <= 1 then
      -- set the output/gradOutput states of initialModule
      local modules = self.initialModule:listModules()
      for i,modula in ipairs(modules) do
         local output, gradInput = modula.output, modula.gradInput
         local output_ = self.initialOutputs[i]
         local gradInput_ = self.initialGradInputs[i] or recursiveClone(gradInput)
         modula.output = output_
         modula.gradInput = gradInput_
      end
      
      -- backward propagate through first step
      local input = self.inputs[1]
      local gradOutput = self.gradOutputs[1]
      if gradInput then
         gradOutput:add(gradInput)
      end
      gradInput = self.initialModule:updateGradInput(input, gradOutput)
      
      for i,modula in ipairs(modules) do
         self.initialGradInputs[i] = modula.gradInput
      end
   end
   
   return gradInput
end

function Recurrent:accGradParametersThroughTime()
   local rho = math.min(self.rho, self.step-1)
   local stop = self.step - rho
   for step=self.step-1,math.max(stop,2),-1 do
      -- set the output/gradOutput states of current Module
      local modules = self.recurrentModule:listModules()
      local recurrentOutputs = self.recurrentOutputs[step]
      local recurrentGradInputs = self.recurrentGradInputs[step]
      
      for i,modula in ipairs(modules) do
         local output, gradInput = modula.output, modula.gradInput
         local output_ = recurrentOutputs[i]
         local gradInput_ = recurrentGradInputs[i]
         assert(output_, "accGradParametersThroughTime should be preceded by updateOutput")
         assert(gradInput_, "accGradParametersThroughTime should be preceded by updateGradInputThroughTime")
         modula.output = output_
         modula.gradInput = gradInput_
      end
      
      -- backward propagate through this step
      local input = self.inputs[step]
      local output = self.outputs[step-1]
      local gradOutput = self.gradOutputs[step]

      local scale = self.scales[step]
      self.recurrentModule:accGradParameters({input, output}, gradOutput, scale/rho)
      
   end
   
   if stop <= 1 then
      -- set the output/gradOutput states of initialModule
      local modules = self.initialModule:listModules()
      for i,modula in ipairs(modules) do
         local output, gradInput = modula.output, modula.gradInput
         local output_ = self.initialOutputs[i]
         local gradInput_ = self.initialGradInputs[i] 
         modula.output = output_
         modula.gradInput = gradInput_
      end
         
      -- backward propagate through first step
      local input = self.inputs[1]
      local gradOutput = self.gradOutputs[1]
      local scale = self.scales[1]
      self.initialModule:accGradParameters(input, gradOutput, scale/rho)
      
      -- startModule's gradParams shouldn't be step-averaged
      -- as it is used only once. So un-step-average it
      local params, gradParams = self.startModule:parameters()
      if gradParams then
         for i,gradParam in ipairs(gradParams) do
            gradParam:mul(rho)
         end
      end
   end
   
   self.gradParametersAccumulated = true
   return gradInput
end

function Recurrent:accUpdateGradParametersThroughTime(lr)
   local rho = math.min(self.rho, self.step-1)
   local stop = self.step - rho
   for step=self.step-1,math.max(stop,2),-1 do
      -- set the output/gradOutput states of current Module
      local modules = self.recurrentModule:listModules()
      local recurrentOutputs = self.recurrentOutputs[step]
      local recurrentGradInputs = self.recurrentGradInputs[step]
      
      for i,modula in ipairs(modules) do
         local output, gradInput = modula.output, modula.gradInput
         local output_ = recurrentOutputs[i]
         local gradInput_ = recurrentGradInputs[i]
         assert(output_, "accGradParametersThroughTime should be preceded by updateOutput")
         assert(gradInput_, "accGradParametersThroughTime should be preceded by updateGradInputThroughTime")
         modula.output = output_
         modula.gradInput = gradInput_
      end
      
      -- backward propagate through this step
      local input = self.inputs[step]
      local output = self.outputs[step-1]
      local gradOutput = self.gradOutputs[step]

      local scale = self.scales[step]
      self.recurrentModule:accUpdateGradParameters({input, output}, gradOutput, lr*scale/rho)
   end
   
   if stop <= 1 then
      -- set the output/gradOutput states of initialModule
      local modules = self.initialModule:listModules()
      for i,modula in ipairs(modules) do
         local output, gradInput = modula.output, modula.gradInput
         local output_ = self.initialOutputs[i]
         local gradInput_ = self.initialGradInputs[i] 
         modula.output = output_
         modula.gradInput = gradInput_
      end
      
      -- backward propagate through first step
      local input = self.inputs[1]
      local gradOutput = self.gradOutputs[1]
      local scale = self.scales[1]
      self.inputModule:accUpdateGradParameters(input, self.startModule.gradInput, lr*scale/rho)
      -- startModule's gradParams shouldn't be step-averaged as it is used only once.
      self.startModule:accUpdateGradParameters(self.inputModule.output, self.transferModule.gradInput, lr*scale)
   end
   
   return gradInput
end

function Recurrent:updateParameters(learningRate)
   if self.gradParametersAccumulated then
      for i=1,#self.modules do
         self.modules[i]:updateParameters(learningRate)
      end
   else
      self:backwardUpdateThroughTime(learningRate)
   end
end

-- goes hand in hand with the next method : forget()
function Recurrent:recycle()
   -- +1 is to skip initialModule
   if self.step > self.rho + 1 then
      assert(self.recurrentOutputs[self.step] == nil)
      assert(self.recurrentOutputs[self.step-self.rho] ~= nil)
      self.recurrentOutputs[self.step] = self.recurrentOutputs[self.step-self.rho]
      self.recurrentGradInputs[self.step] = self.recurrentGradInputs[self.step-self.rho]
      self.recurrentOutputs[self.step-self.rho] = nil
      self.recurrentGradInputs[self.step-self.rho] = nil
      -- need to keep rho+1 of these
      self.outputs[self.step] = self.outputs[self.step-self.rho-1] 
      self.outputs[self.step-self.rho-1] = nil
   end
   if self.step > self.rho then
      assert(self.inputs[self.step] == nil)
      assert(self.inputs[self.step-self.rho] ~= nil)
      self.inputs[self.step] = self.inputs[self.step-self.rho] 
      self.gradOutputs[self.step] = self.gradOutputs[self.step-self.rho] 
      self.inputs[self.step-self.rho] = nil
      self.gradOutputs[self.step-self.rho] = nil
      self.scales[self.step-self.rho] = nil
   end
end

function Recurrent:forget()

   if self.train ~= false then
      -- bring all states back to the start of the sequence buffers
      local lastStep = self.step - 1
      
      if lastStep > self.rho + 1 then
         local i = 2
         for step = lastStep-self.rho+1,lastStep do
            self.recurrentOutputs[i] = self.recurrentOutputs[step]
            self.recurrentGradInputs[i] = self.recurrentGradInputs[step]
            self.recurrentOutputs[step] = nil
            self.recurrentGradInputs[step] = nil
            -- we keep rho+1 of these : outputs[k]=outputs[k+rho+1]
            self.outputs[i-1] = self.outputs[step]
            self.outputs[step] = nil
            i = i + 1
         end
         
      end
      
      if lastStep > self.rho then
         local i = 1
         for step = lastStep-self.rho+1,lastStep do
            self.inputs[i] = self.inputs[step]
            self.gradOutputs[i] = self.gradOutputs[step]
            self.inputs[step] = nil
            self.gradOutputs[step] = nil
            self.scales[step] = nil
            i = i + 1
         end

      end
   end
   
   -- forget the past inputs; restart from first step
   self.step = 1
end

function Recurrent:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = torch.type(self)
   str = str .. ' {' .. line .. tab .. '[{input(t), output(t-1)}'
   for i=1,3 do
      str = str .. next .. '(' .. i .. ')'
   end
   str = str .. next .. 'output(t)]'
   
   local tab = '  '
   local line = '\n  '
   local next = '  |`-> '
   local ext = '  |    '
   local last = '   ... -> '
   str = str .. line ..  '(1): ' .. ' {' .. line .. tab .. 'input(t)'
   str = str .. line .. tab .. next .. '(t==0): ' .. tostring(self.startModule):gsub('\n', '\n' .. tab .. ext)
   str = str .. line .. tab .. next .. '(t~=0): ' .. tostring(self.inputModule):gsub('\n', '\n' .. tab .. ext)
   str = str .. line .. tab .. 'output(t-1)'
   str = str .. line .. tab .. next .. tostring(self.feedbackModule):gsub('\n', line .. tab .. ext)
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   str = str .. line .. tab .. '(' .. 2 .. '): ' .. tostring(self.mergeModule):gsub(line, line .. tab)
   str = str .. line .. tab .. '(' .. 3 .. '): ' .. tostring(self.transferModule):gsub(line, line .. tab)
   str = str .. line .. '}'
   return str
end
