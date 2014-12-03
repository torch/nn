local Module = torch.class('nn.Module')

function Module:__init()
   self.gradInput = torch.Tensor()
   self.output = torch.Tensor()
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

local function recursiveType(param, type_str)
   if torch.type(param) == 'table' then
      for i = 1, #param do
         param[i] = recursiveType(param[i], type_str)
      end
   else
      if torch.typename(param) and 
        torch.typename(param):find('torch%..+Tensor') then
         param = param:type(type_str)
      end
   end
   return param
end

function Module:type(type)
   assert(type, 'Module: must provide a type to convert to')
   -- find all tensors and convert them
   for key,param in pairs(self) do
      -- Many modules (like CDivTable) have output or gradInput fields which
      -- are table's of tensors.  To be general we need to recursively
      -- cast fields that may be nested tables.
      if key ~= 'modules' then
        self[key] = recursiveType(self[key], type)
      end
   end
   -- find submodules in classic containers 'modules'
   if self.modules then
      for _,module in ipairs(self.modules) do
         module:type(type)
      end
   end
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

function Module:getParameters()
   -- get parameters
   local parameters,gradParameters = self:parameters()

   local function storageInSet(set, storage)
      local storageAndOffset = set[torch.pointer(storage)]
      if storageAndOffset == nil then
          return nil
      end
      local _, offset = unpack(storageAndOffset)
      return offset
   end

   -- this function flattens arbitrary lists of parameters,
   -- even complex shared ones
   local function flatten(parameters)
      if not parameters or #parameters == 0 then
         return torch.Tensor()
      end
      local Tensor = parameters[1].new

      local storages = {}
      local nParameters = 0
      for k = 1,#parameters do
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

      local maskParameters=  flatParameters:float():clone()
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
         local k, v = unpack(storageAndOffset)
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
   local flatGradParameters = flatten(gradParameters)

   -- return new flat vector that contains all discrete parameters
   return flatParameters, flatGradParameters
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

