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
   return self:updateOutput(input, target)
end

function Module:backward(input, gradOutput)
   self:updateGradInput(input, gradOutput)
   self:accGradParameters(input, gradOutput)
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

function Module:share(mlp, ...)
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

function Module:type(type)
   -- find all tensors and convert them
   for key,param in pairs(self) do
      if torch.typename(param) and torch.typename(param):find('torch%..+Tensor') then
         self[key] = param:type(type)
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

function Module:getParameters()
   -- get parameters
   local parameters,gradParameters = self:parameters()

   -- this function flattens arbitrary lists of parameters,
   -- even complex shared ones
   local function flatten(parameters)
      -- already flat ?
      local flat = true
      for k = 2,#parameters do
         if parameters[k]:storage() ~= parameters[k-1]:storage() then
            flat = false
            break
         end
      end
      if flat then
         local nParameters = 0
         for k,param in ipairs(parameters) do
            nParameters = nParameters + param:nElement()
         end
         local flatParameters = parameters[1].new(parameters[1]:storage())
         if nParameters ~= flatParameters:nElement() then
            error('flattenParameters(): weird parameters')
         end
         return flatParameters
      end
      -- compute offsets of each parameter
      local offsets = {}
      local sizes = {}
      local strides = {}
      local elements = {}
      local storageOffsets = {}
      local params = {}
      local nParameters = 0
      for k,param in ipairs(parameters) do
         table.insert(offsets, nParameters+1)
         table.insert(sizes, param:size())
         table.insert(strides, param:stride())
         table.insert(elements, param:nElement())
         table.insert(storageOffsets, param:storageOffset())
         local isView = false
         for i = 1,k-1 do
            if param:storage() == parameters[i]:storage() then
               offsets[k] = offsets[i]
               if storageOffsets[k] ~= storageOffsets[i] or elements[k] ~= elements[i] then
                  error('flattenParameters(): cannot flatten shared weights with different structures')
               end
               isView = true
               break
            end
         end
         if not isView then
            nParameters = nParameters + param:nElement()
         end
      end
      -- create flat vector
      local flatParameters = parameters[1].new(nParameters)
      local storage = flatParameters:storage()
      -- reallocate all parameters in flat vector
      for i = 1,#parameters do
         local data = parameters[i]:clone()
         parameters[i]:set(storage, offsets[i], elements[i]):resize(sizes[i],strides[i]):copy(data)
         data = nil
         collectgarbage()
      end
      -- cleanup
      collectgarbage()
      -- return flat param
      return flatParameters
   end

   -- flatten parameters and gradients
   local flatParameters = flatten(parameters)
   local flatGradParameters = flatten(gradParameters)

   -- return new flat vector that contains all discrete parameters
   return flatParameters, flatGradParameters
end
