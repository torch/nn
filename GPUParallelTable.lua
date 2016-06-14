local GPUParallelTable, parent = torch.class('nn.GPUParallelTable', 'nn.ParallelTable')

function GPUParallelTable:__init()
   parent.__init(self)
   self.modules = {}
   self.devices = {}
   self.outdevices = {}
   self.output = {}
   self.gradInput = {}
end

function GPUParallelTable:add(module, device, outdevice)
   assert(torch.isTypeOf(module, 'nn.Module'))
   assert(torch.type(device) == 'number')
   table.insert(self.modules, module)
   table.insert(self.devices, device)
   assert(#self.modules == #self.devices)
   self.outdevices[#self.modules] = outdevice or device
   return self
end

function GPUParallelTable:updateOutput(input)
   if self._type == 'torch.CudaTensor' then
      -- send input to appriopriate device, if necessary (blocking, so serial)
      self._input = nn.GPU.recursiveSetDeviceAs(self._input, input, self.devices)
      
      -- then forward in parallel across devices (if module is non-blocking, happens concurrently, yay!)
      local output = {}
      for i,module in ipairs(self.modules) do
         local device = self.devices[i]
         output[i] = cutorch.withDevice(device, function()
            return self:rethrowErrors(module, i, 'updateOutput', self._input[i])
         end)
      end
      
      -- send output to appriopriate device, if necessary (blocking, so serial)
      self.output = nn.GPU.recursiveSetDeviceAs(self.output, output, self.outdevices)
   else
      parent.updateOutput(self, input) 
   end
   
   return self.output
end

function GPUParallelTable:updateGradInput(input, gradOutput)
   if self._type == 'torch.CudaTensor' then
      -- send gradOutput to appriopriate device, if necessary (blocking, so serial)
      self._gradOutput = nn.GPU.recursiveSetDeviceAs(self._gradOutput, gradOutput, self.devices)
      
      -- then updateGradInput in parallel across devices (if module is non-blocking, happens concurrently)
      local gradInput = {}
      for i,module in ipairs(self.modules) do
         local device = self.devices[i]
         gradInput[i] = cutorch.withDevice(device, function()
            return self:rethrowErrors(module, i, 'updateGradInput', self._input[i], self._gradOutput[i])
         end)
      end
      
      -- send gradInput to appriopriate device, if necessary (blocking, so serial)
      self.gradInput = nn.GPU.recursiveSetDeviceAs(self.gradInput, gradInput, self.input)
   else
      parent.updateGradInput(self, input, gradOutput) 
   end
   
   return self.gradInput
end

function GPUParallelTable:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   
   if self._type == 'torch.CudaTensor' then
      -- accGradParameters in parallel across devices (if module is non-blocking, happens concurrently)
      for i,module in ipairs(self.modules) do
         cutorch.withDevice(self.devices[i], function()
            self:rethrowErrors(module, i, 'accGradParameters', self._input[i], self._gradOutput[i], scale)
         end)
      end
   else
      parent.accGradParameters(self, input, gradOutput, scale) 
   end
end

function GPUParallelTable:accUpdateGradParameters(input, gradOutput, lr)
   lr = lr or 1
   
   if self._type == 'torch.CudaTensor' then
      -- accUpdateGradParameters in parallel across devices (if module is non-blocking, happens concurrently)
      for i,module in ipairs(self.modules) do
         cutorch.withDevice(self.devices[i], function()
            self:rethrowErrors(module, i, 'accUpdateGradParameters', self._input[i], self._gradOutput[i], lr)
         end)
      end
   else
      parent.accUpdateGradParameters(self, input, gradOutput, lr) 
   end
end

function GPUParallelTable:type(type, typecache)
   self.output = {}
   self.gradInput = {}
   self._input = {}
   self._gradOutput = {}
   if type and type == 'torch.CudaTensor' then
      for i,module in ipairs(self.modules) do
         local device = self.devices[i]
         cutorch.withDevice(self.device, function() module:type(type, typecache) end)
         self.modules[i] = cutorch.withDevice(device, function() 
            return nn.GPU.recursiveModuleDevice(module, device)
         end)
      end
      self._type = type
   else
      parent.type(self, type, typecache)
   end
   return self
end


-- TODO : wrap all the other fucking methods.
