----------------------------------------------------------------------
-- hessian.lua: this file appends extra methods to modules in nn,
-- to estimate diagonal elements of the Hessian. This is useful
-- to condition learning rates individually.
----------------------------------------------------------------------
nn.hessian = {}

----------------------------------------------------------------------
-- Hessian code is still experimental,
-- and deactivated by default
----------------------------------------------------------------------
function nn.hessian.enable()

   local function accDiagHessianParameters(module, input, diagHessianOutput, gw, hw)
      if #gw ~= #hw then
         error('Number of gradients is nto equal to number of hessians')
      end
      module.inputSq = module.inputSq or input.new()
      module.inputSq:resizeAs(input)
      torch.cmul(module.inputSq, input, input)
      -- replace gradients with hessian
      for i=1,#gw do
         local gwname = gw[i]
         local hwname = hw[i]
         local gwval = module[gwname]
         local hwval = module[hwname]
         if hwval == nil then
            module[hwname] = gwval.new():resizeAs(gwval)
            hwval = module[hwname]
         end
         module[gwname] = hwval
         module[hwname] = gwval
      end
      module.accGradParameters(module, module.inputSq, diagHessianOutput, 1)
      -- put back gradients
      for i=1,#gw do
         local gwname = gw[i]
         local hwname = hw[i]
         local gwval = module[gwname]
         local hwval = module[hwname]
         module[gwname] = hwval
         module[hwname] = gwval
      end
   end
   nn.hessian.accDiagHessianParameters = accDiagHessianParameters

   local function updateDiagHessianInput(module, input, diagHessianOutput, w, wsq)
      if #w ~= #wsq then
         error('Number of weights is not equal to number of weights squares')
      end
      module.diagHessianInput = module.diagHessianInput or input.new()
      module.diagHessianInput:resizeAs(input)

      local gi = module.gradInput
      module.gradInput = module.diagHessianInput
      for i=1,#w do
         local wname = w[i]
         local wsqname = wsq[i]
         local wval = module[wname]
         local wsqval = module[wsqname]
         if wsqval == nil then
            module[wsqname] = wval.new()
            wsqval = module[wsqname]
         end
         wsqval:resizeAs(wval)
         torch.cmul(wsqval, wval, wval)
         module[wsqname] = wval
         module[wname] = wsqval
      end
      module.updateGradInput(module,input,diagHessianOutput)
      for i=1,#w do
         local wname = w[i]
         local wsqname = wsq[i]
         local wval = module[wname]
         local wsqval = module[wsqname]
         module[wname] = wsqval
         module[wsqname] = wval
      end
      module.gradInput = gi
   end
   nn.hessian.updateDiagHessianInput = updateDiagHessianInput

   local function updateDiagHessianInputPointWise(module, input, diagHessianOutput)
      local tdh = diagHessianOutput.new():resizeAs(diagHessianOutput):fill(1)
      updateDiagHessianInput(module,input,tdh,{},{})
      module.diagHessianInput:cmul(module.diagHessianInput)
      module.diagHessianInput:cmul(diagHessianOutput)
   end
   nn.hessian.updateDiagHessianInputPointWise = updateDiagHessianInputPointWise

   local function initDiagHessianParameters(module,gw,hw)
      module.diagHessianInput = module.diagHessianInput or module.gradInput.new();
      for i=1,#gw do
         module[hw[i]] = module[hw[i]] or module[gw[i]].new():resizeAs(module[gw[i]])
      end
   end
   nn.hessian.initDiagHessianParameters = initDiagHessianParameters

   ----------------------------------------------------------------------
   -- Module
   ----------------------------------------------------------------------
   function nn.Module.updateDiagHessianInput(self, input, diagHessianOutput)
      error(torch.typename(self) .. ':updateDiagHessianInput() is undefined')
   end

   function nn.Module.accDiagHessianParameters(self, input, diagHessianOutput)
   end

   function nn.Module.initDiagHessianParameters()
   end

   ----------------------------------------------------------------------
   -- Sequential
   ----------------------------------------------------------------------
   function nn.Sequential.initDiagHessianParameters(self)
      for i=1,#self.modules do
         self.modules[i]:initDiagHessianParameters()
      end
   end

   function nn.Sequential.updateDiagHessianInput(self, input, diagHessianOutput)
      local currentDiagHessianOutput = diagHessianOutput
      local currentModule = self.modules[#self.modules]
      for i=#self.modules-1,1,-1 do
         local previousModule = self.modules[i]
         currentDiagHessianOutput = currentModule:updateDiagHessianInput(previousModule.output, currentDiagHessianOutput)
         currentModule = previousModule
      end
      currentDiagHessianOutput = currentModule:updateDiagHessianInput(input, currentDiagHessianOutput)
      self.diagHessianInput = currentDiagHessianOutput
      return currentDiagHessianOutput
   end

   function nn.Sequential.accDiagHessianParameters(self, input, diagHessianOutput)
      local currentDiagHessianOutput = diagHessianOutput
      local currentModule = self.modules[#self.modules]
      for i=#self.modules-1,1,-1 do
         local previousModule = self.modules[i]
         currentModule:accDiagHessianParameters(previousModule.output, currentDiagHessianOutput)
         currentDiagHessianOutput = currentModule.diagHessianInput
         currentModule = previousModule
      end
      currentModule:accDiagHessianParameters(input, currentDiagHessianOutput)
   end

   ----------------------------------------------------------------------
   -- Criterion
   ----------------------------------------------------------------------
   function nn.Criterion.updateDiagHessianInput(self, input, diagHessianOutput)
      error(torch.typename(self) .. ':updateDiagHessianInput() is undefined')
   end

   ----------------------------------------------------------------------
   -- MSECriterion
   ----------------------------------------------------------------------
   function nn.MSECriterion.updateDiagHessianInput(self, input, target)
      self.diagHessianInput = self.diagHessianInput or input.new()
      local val = 2
      if self.sizeAverage then
         val = val / input:nElement()
      end
      self.diagHessianInput:resizeAs(input):fill(val)
      return self.diagHessianInput
   end

   ----------------------------------------------------------------------
   -- Linear
   ----------------------------------------------------------------------
   function nn.Linear.updateDiagHessianInput(self, input, diagHessianOutput)
      updateDiagHessianInput(self, input, diagHessianOutput, {'weight'}, {'weightSq'})
      return self.diagHessianInput
   end

   function nn.Linear.accDiagHessianParameters(self, input, diagHessianOutput)
      accDiagHessianParameters(self,input, diagHessianOutput, {'gradWeight','gradBias'}, {'diagHessianWeight','diagHessianBias'})
   end

   function nn.Linear.initDiagHessianParameters(self)
      initDiagHessianParameters(self,{'gradWeight','gradBias'},{'diagHessianWeight','diagHessianBias'})
   end

   ----------------------------------------------------------------------
   -- SpatialConvolution
   ----------------------------------------------------------------------
   function nn.SpatialConvolution.updateDiagHessianInput(self, input, diagHessianOutput)
      updateDiagHessianInput(self, input, diagHessianOutput, {'weight'}, {'weightSq'})
      return self.diagHessianInput
   end

   function nn.SpatialConvolution.accDiagHessianParameters(self, input, diagHessianOutput)
      accDiagHessianParameters(self,input, diagHessianOutput, {'gradWeight'}, {'diagHessianWeight'})
   end

   function nn.SpatialConvolution.initDiagHessianParameters(self)
      initDiagHessianParameters(self,{'gradWeight'},{'diagHessianWeight'})
   end

   ----------------------------------------------------------------------
   -- SpatialConvolutionMap
   ----------------------------------------------------------------------
   function nn.SpatialConvolutionMap.updateDiagHessianInput(self, input, diagHessianOutput)
      updateDiagHessianInput(self, input, diagHessianOutput, {'weight','bias'}, {'weightSq','biasSq'})
      return self.diagHessianInput
   end

   function nn.SpatialConvolutionMap.accDiagHessianParameters(self, input, diagHessianOutput)
      accDiagHessianParameters(self,input, diagHessianOutput, {'gradWeight','gradBias'}, {'diagHessianWeight','diagHessianBias'})
   end

   function nn.SpatialConvolutionMap.initDiagHessianParameters(self)
      initDiagHessianParameters(self,{'gradWeight','gradBias'},{'diagHessianWeight','diagHessianBias'})
   end

   ----------------------------------------------------------------------
   -- Tanh
   ----------------------------------------------------------------------
   function nn.Tanh.updateDiagHessianInput(self, input, diagHessianOutput)
      updateDiagHessianInputPointWise(self, input, diagHessianOutput)
      return self.diagHessianInput
   end

   ----------------------------------------------------------------------
   -- Square
   ----------------------------------------------------------------------
   function nn.Square.updateDiagHessianInput(self, input, diagHessianOutput)
      updateDiagHessianInputPointWise(self, input, diagHessianOutput)
      return self.diagHessianInput
   end

   ----------------------------------------------------------------------
   -- Sqrt
   ----------------------------------------------------------------------
   function nn.Sqrt.updateDiagHessianInput(self, input, diagHessianOutput)
      updateDiagHessianInputPointWise(self, input, diagHessianOutput)
      return self.diagHessianInput
   end

   ----------------------------------------------------------------------
   -- Reshape
   ----------------------------------------------------------------------
   function nn.Reshape.updateDiagHessianInput(self, input, diagHessianOutput)
      self.diagHessianInput = self.diagHessianInput or input.new()
      diagHessianOutput = diagHessianOutput:contiguous()
      self.diagHessianInput:set(diagHessianOutput):resizeAs(input)
      return self.diagHessianInput
   end

   ----------------------------------------------------------------------
   -- Parameters manipulation:
   -- we modify these functions such that they return hessian coefficients
   ----------------------------------------------------------------------
   function nn.Module.parameters(self)
      if self.weight and self.bias then
         return {self.weight, self.bias}, {self.gradWeight, self.gradBias}, {self.diagHessianWeight, self.diagHessianBias}
      elseif self.weight then
         return {self.weight}, {self.gradWeight}, {self.diagHessianWeight}
      elseif self.bias then
         return {self.bias}, {self.gradBias}, {self.diagHessianBias}
      else
         return
      end
   end

   function nn.Module.getParameters(self)
      -- get parameters
      local parameters,gradParameters,hessianParameters = self:parameters()

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
      local flatHessianParameters
      if hessianParameters and hessianParameters[1] then
         flatHessianParameters = flatten(hessianParameters)
      end

      -- return new flat vector that contains all discrete parameters
      return flatParameters, flatGradParameters, flatHessianParameters
   end

   function nn.Sequential.parameters(self)
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
      local ggw = {}
      for i=1,#self.modules do
         local mw,mgw,mggw = self.modules[i]:parameters()
         if mw then
            tinsert(w,mw)
            tinsert(gw,mgw)
            tinsert(ggw,mggw)
         end
      end
      return w,gw,ggw
   end

   ----------------------------------------------------------------------
   -- Avoid multiple calls to enable()
   ----------------------------------------------------------------------
   function nn.hessian.enable()
   end
end
