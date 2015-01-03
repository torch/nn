local Module = torch.class('nn.Module')

function Module:__init()
   self.gradInput = torch.Tensor()
   self.output = torch.Tensor()
   self._type = self.output:type()
end

function Module:parameters()
   if self.weight and self.bias then
      return {self.weight, self.bias}, {self.gradWeight, self.gradBias}
   elseif self.weight then
      return {self.weight}, {self.gradWeight}
   elseif self.bias then
      return {self.bias}, {self.gradBias}
   else
      return
   end
end

function Module:updateOutput(input)
   return self.output
end

function Module:forward(input)
   return self:updateOutput(input)
end

function Module:backward(input, gradOutput, scale)
   scale = scale or 1
   self:updateGradInput(input, gradOutput)
   self:accGradParameters(input, gradOutput, scale)
   return self.gradInput
end

function Module:backwardUpdate(input, gradOutput, lr)
   self:updateGradInput(input, gradOutput)
   self:accUpdateGradParameters(input, gradOutput, lr)
   return self.gradInput
end

function Module:updateGradInput(input, gradOutput)
   return self.gradInput
end

function Module:accGradParameters(input, gradOutput, scale)
end

function Module:accUpdateGradParameters(input, gradOutput, lr)
   local gradWeight = self.gradWeight
   local gradBias = self.gradBias
   self.gradWeight = self.weight
   self.gradBias = self.bias
   self:accGradParameters(input, gradOutput, -lr)
   self.gradWeight = gradWeight
   self.gradBias = gradBias
end

function Module:sharedAccUpdateGradParameters(input, gradOutput, lr)
   if self:parameters() then
      self:zeroGradParameters()
      self:accGradParameters(input, gradOutput, 1)
      self:updateParameters(lr)
   end
end

function Module:zeroGradParameters()
   local _,gradParams = self:parameters()
   if gradParams then
      for i=1,#gradParams do
         gradParams[i]:zero()
      end
   end
end

function Module:updateParameters(learningRate)
   local params, gradParams = self:parameters()
   if params then
      for i=1,#params do
         params[i]:add(-learningRate, gradParams[i])
      end
   end
end

function Module:training()
   self.train = true
end

function Module:evaluate()
   self.train = false
end

function Module:share(mlp, ...)
   local arg = {...}
   for i,v in ipairs(arg) do
      if self[v] ~= nil then
         self[v]:set(mlp[v])
         self.accUpdateGradParameters = self.sharedAccUpdateGradParameters
         mlp.accUpdateGradParameters = mlp.sharedAccUpdateGradParameters
      end
   end
   return self
end

function Module:clone(...)
   local f = torch.MemoryFile("rw"):binary()
   f:writeObject(self)
   f:seek(1)
   local clone = f:readObject()
   f:close()
   if select('#',...) > 0 then
      clone:share(self,...)
   end
   return clone
end

function Module:_freeCaches()
   -- nop. See invocation in Module:type()
end

function Module:type(type)
   if not type then
      return self._type
   end
   -- Some modules keep around CudaTensors and other data that's
   -- impossible to serialize as scratchpads. Invokes the module-
   -- dependent logic to clear these caches.
   self:_freeCaches()

   -- find all tensors and convert them
   for key,param in pairs(self) do
      if self.gpu_assignments and key == 'modules' then
	 local current_gpuid = cutorch.getDevice()
         for i, module in ipairs(self.modules) do
            cutorch.setDevice(self.gpu_assignments[i])
            module:float():type(type)
         end
         cutorch.setDevice(current_gpuid)
      else
	 self[key] = nn.utils.recursiveType(param, type)
      end
   end
   self._type = type
   return self
end

function Module:float()
   return self:type('torch.FloatTensor')
end

function Module:double()
   return self:type('torch.DoubleTensor')
end

function Module:cuda()
   return self:type('torch.CudaTensor')
end

function Module:reset()
end

-- This function is not easy to understand. It works as follows:
--
-- - gather all parameter tensors for this module (and children);
--   count all parameter values (floats)
-- - create one ginormous memory area (Storage object) with room for all
--   parameters
-- - remap each parameter tensor to point to an area within the ginormous
--   Storage, and copy it there
--
-- It has the effect of making all parameters point to the same memory area,
-- which is then returned.
--
-- The purpose is to allow operations over all parameters (such as momentum
-- updates and serialization), but it assumes that all parameters are of
-- the same type (and, in the case of CUDA, on the same device), which
-- is not always true. Use for_each() to iterate over this module and
-- children instead.
--
-- TODO: This logically belongs to torch.Tensor, not nn.
function Module._gather(tensors)
   local function storageInSet(set, storage)
      local storageAndOffset = set[torch.pointer(storage)]
      if storageAndOffset == nil then
         return nil
      end
      local _, offset = table.unpack(storageAndOffset)
      return offset
   end

   -- this function flattens arbitrary lists of parameters,
   -- even complex shared ones
   local function flatten(parameters)
      if not parameters or #parameters == 0 then
         return torch.Tensor()
      end
      local Tensor = parameters[1].new
      local dtype = parameters[1]:type()

      local storages = {}
      local nParameters = 0
      for k = 1,#parameters do
         if parameters[k]:type() ~= dtype then
            error("Inconsistent parameter types. " .. parameters[k]:type() ..
                  " ~= " .. dtype)
         end
         local storage = parameters[k]:storage()
         if not storageInSet(storages, storage) then
            storages[torch.pointer(storage)] = {storage, nParameters}
            nParameters = nParameters + storage:size()
         end
      end

      local flatParameters = Tensor(nParameters):fill(1)
      local flatStorage = flatParameters:storage()

      for k = 1,#parameters do
         local storageOffset = storageInSet(storages, parameters[k]:storage())
         parameters[k]:set(flatStorage,
                           storageOffset + parameters[k]:storageOffset(),
                           parameters[k]:size(),
                           parameters[k]:stride())
         parameters[k]:zero()
      end

      local maskParameters = flatParameters:float():clone()
      local cumSumOfHoles = flatParameters:float():cumsum(1)
      local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
      local flatUsedParameters = Tensor(nUsedParameters)
      local flatUsedStorage = flatUsedParameters:storage()

      for k = 1,#parameters do
         local offset = cumSumOfHoles[parameters[k]:storageOffset()]
         parameters[k]:set(flatUsedStorage,
                           parameters[k]:storageOffset() - offset,
                           parameters[k]:size(),
                           parameters[k]:stride())
      end

      for _, storageAndOffset in pairs(storages) do
         local k, v = table.unpack(storageAndOffset)
         flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
      end

      if cumSumOfHoles:sum() == 0 then
         flatUsedParameters:copy(flatParameters)
      else
         local counter = 0
         for k = 1,flatParameters:nElement() do
            if maskParameters[k] == 0 then
               counter = counter + 1
               flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
            end
         end
         assert (counter == nUsedParameters)
      end
      return flatUsedParameters
   end

   -- flatten parameters and gradients
   local flatParameters = flatten(parameters)
   collectgarbage()
   local flatGradParameters = flatten(gradParameters)
   collectgarbage()

   -- return new flat vector that contains all discrete parameters
   return flatten(tensors)
end

function Module:getParameters()
   -- get parameters
   local parameters,gradParameters = self:parameters()
   return Module._gather(parameters), Module._gather(gradParameters)
end

-- Get the params of the module separated by device. Vanilla nn,
-- without cunn, has all parameters in device 0.
function Module:getParametersByDevice()
    local p, gp = self:getParameters()
    local tp = { }
    local tgp = { }
    tp[0] = p
    tgp[0] = gp
    return p, gp
end

function Module:__call__(input, gradOutput)
    self:forward(input)
    if gradOutput then
        self:backward(input, gradOutput)
        return self.output, self.gradInput
    else
        return self.output
    end
end

-- Run a callback (called with the module as an argument) in preorder over this
-- module and its children. If flag_name is specified, we use it as an
-- indicator to mark modules that have already been processed; if you call
-- for_each again with the same flag_name, the processed modules (and all
-- of their children) will be skipped.
function Module:for_each(callback, flag_name)
    if flag_name then
        flag_name = '_for_each_' .. flag_name
    end
    self:_for_each(callback, flag_name)
end

function Module:_for_each(callback, flag_name)
    if flag_name and self[flag_name] then
        return
    end
    callback(self)
    if flag_name then
        self[flag_name] = true
    end
    if not self.modules then
        return
    end
    if self.gpu_assignments then
        local current_gpuid = cutorch.getDevice()
        for i, module in ipairs(self.modules) do
            cutorch.setDevice(self.gpu_assignments[i])
            module:_for_each(callback, flag_name)
        end
        cutorch.setDevice(current_gpuid)
    else
        for _, module in ipairs(self.modules) do
            module:_for_each(callback, flag_name)
        end
    end
end

function Module:findModules(typename, container)
  container = container or self
  local nodes = {}
  local containers = {}
  local mod_type = torch.typename(self)
  if mod_type == typename then
    nodes[#nodes+1] = self
    containers[#containers+1] = container
  end
  -- Recurse on nodes with 'modules'
  if (self.modules ~= nil) then
    if (torch.type(self.modules) == 'table') then
      for i = 1, #self.modules do
        local child = self.modules[i]
        local cur_nodes, cur_containers =
          child:findModules(typename, self)
        assert(#cur_nodes == #cur_containers,
          'Internal error: incorrect return length')  -- This shouldn't happen
        -- add the list items from our child to our list (ie return a
        -- flattened table of the return nodes).
        for j = 1, #cur_nodes do
          nodes[#nodes+1] = cur_nodes[j]
          containers[#containers+1] = cur_containers[j]
        end
      end
    end
  end
  return nodes, containers
end

-- returns a list of modules
function Module:listModules()
   local function tinsert(to, from)
      if torch.type(from) == 'table' then
         for i=1,#from do
            tinsert(to,from[i])
         end
      else
         table.insert(to,from)
      end
   end
   -- include self first
   local modules = {self}
   if self.modules then
      for i=1,#self.modules do
         local modulas = self.modules[i]:listModules()
         if modulas then
            tinsert(modules,modulas)
         end
      end
   end
   return modules
end
