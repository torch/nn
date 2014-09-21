------------------------------------------------------------------------
--[[ SpatialConcat ]]--
-- Concatenates the output of Spatial modules along the depth dimension
-- (nOutputFrame). This is used to implement the DepthConcat layer 
-- of the Going deeper with convolutions paper :
-- http://arxiv.org/pdf/1409.4842v1.pdf
-- The normal Concat Module can't be used since the spatial dimensions
-- tensors to be concatenated may have different values. To deal with
-- this, we select the largest spatial dimensions and add zero-padding
-- around the smaller tensors.
------------------------------------------------------------------------
local SpatialConcat, parent = torch.class('nn.SpatialConcat', 'nn.Concat')

function SpatialConcat:spatialWindow(output, currentOutput, offset)
   local outputWindow = output:narrow(self.dimension, offset, currentOutput:size(self.dimension))
   for dim=1,self.size:size(1) do
      local currentSize = currentOutput:size(dim)
      if dim ~= self.dimension and self.size[dim] ~= currentSize then
         -- 5x5 vs 3x3 -> start = [(5-3)/2] + 1 = 2 (1 pad each side)
         -- 9x9 vs 5x5 -> start = [(9-5)/2] + 1 = 3 (2 pad each side)
         -- 9x9 vs 4x4 -> start = [(9-4)/2] + 1 = 3.5 (2 pad, 3 pad)
         local start = ((self.size[dim] - currentSize) / 2) + 1
         outputWindow = outputWindow:narrow(dim, start, currentSize)
      end
   end
   return outputWindow
end

function SpatialConcat:updateOutput(input)
   local outs = {}
   for i=1,#self.modules do
      local currentOutput = self.modules[i]:updateOutput(input)
      outs[i] = currentOutput
      if i == 1 then
         self.size:resize(currentOutput:dim()):copy(currentOutput:size())
      else
         self.size[self.dimension] = self.size[self.dimension] + currentSize[self.dimension]
         for dim=1,self.size:size(1) do
            if dim ~= self.dimension then
               -- take the maximum size (shouldn't change anything for batch dim)
               self.size[dim] = math.max(self.size[dim], currentOutput:size(dim))
            end
         end
      end
   end
   self.output:resize(self.size):zero() --zero for padding
   
   local offset = 1
   for i,module in ipairs(self.modules) do
      local currentOutput = outs[i]
      local outputWindow = self:spatialWindow(self.output, currentOutput, offset)
      outputWindow:copy(currentOutput)
      offset = offset + currentOutput:size(self.dimension)
   end
   return self.output
end
   
function SpatialConcat:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)

   local offset = 1
   for i,module in ipairs(self.modules) do
      local gradOutputWindow = self:spatialWindow(gradOutput, module.output, offset)
      local currentGradInput = module:updateGradInput(input, gradOutputWindow)
      if i==1 then
         self.gradInput:copy(currentGradInput)
      else
         self.gradInput:add(currentGradInput)
      end
      offset = offset + currentOutput:size(self.dimension)
   end
   return self.gradInput
end

function SpatialConcat:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   local offset = 1
   for i,module in ipairs(self.modules) do
      local gradOutputWindow = self:spatialWindow(gradOutput, module.output, offset)
      local currentGradInput = module:accGradParameters(input, gradOutputWindow, scale)
      offset = offset + currentOutput:size(self.dimension)
   end
end

function SpatialConcat:accUpdateGradParameters(input, gradOutput, lr)
   local offset = 1
   for i,module in ipairs(self.modules) do
      local gradOutputWindow = self:spatialWindow(gradOutput, module.output, offset)
      local currentGradInput = module:accUpdateGradParameters(input, gradOutputWindow, lr)
      offset = offset + currentOutput:size(self.dimension)
   end
end

