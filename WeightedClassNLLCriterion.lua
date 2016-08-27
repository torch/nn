local THNN = require 'nn.THNN'
local WeightedClassNLLCriterion, parent = torch.class('nn.WeightedClassNLLCriterion', 'nn.Criterion')

function WeightedClassNLLCriterion:__init(weights, sizeAverage)
    parent.__init(self)
    if sizeAverage ~= nil then
       self.sizeAverage = sizeAverage
    else
       self.sizeAverage = true
    end

    self.output_tensor = torch.zeros(1)
    self.total_weight_tensor = torch.ones(1)
    self.target = torch.zeros(1):long()
end

function WeightedClassNLLCriterion:__len()
   if (self.weights) then
      return #self.weights
   else
      return 0
   end
end

function WeightedClassNLLCriterion:updateOutput(input, target)
   if type(target) == 'table' then -- has weights
      if type(target[1]) == 'number' then
         if input:type() ~= 'torch.CudaTensor' then
            self.target = self.target:long()
         end
         self.target[1] = target[1]
         self.weights   = target[2]
      elseif target[1]:type() == 'torch.CudaTensor' then
         self.target  = target[1]
         self.weights = target[2]
      else -- torch.Tensor
         self.target  = target[1]:long()
         self.weights = target[2]
      end
   else -- w/o weights
      if type(target) == 'number' then
         if input:type() ~= 'torch.CudaTensor' then
            self.target = self.target:long()
         end
         self.target[1] = target
      elseif target:type() == 'torch.CudaTensor' then
         self.target = target
      else
         self.target = target:long()
      end
      self.weights = nil
   end

   input.THNN.WeightedClassNLLCriterion_updateOutput(
      input:cdata(),
      self.target:cdata(),
      self.output_tensor:cdata(),
      self.sizeAverage,
      THNN.optionalTensor(self.weights),
      self.total_weight_tensor:cdata()
   )
   self.output = self.output_tensor[1]
   return self.output, self.total_weight_tensor[1]
end

function WeightedClassNLLCriterion:updateGradInput(input, target)
   if type(target) == 'table' then -- has weights
      if type(target[1]) == 'number' then
         if input:type() ~= 'torch.CudaTensor' then
            self.target = self.target:long()
         end
         self.target[1] = target[1]
         self.weights   = target[2]
      elseif target[1]:type() == 'torch.CudaTensor' then
         self.target  = target[1]
         self.weights = target[2]
      else -- torch.Tensor
         self.target  = target[1]:long()
         self.weights = target[2]
      end
   else -- w/o weights
      if type(target) == 'number' then
         if input:type() ~= 'torch.CudaTensor' then
            self.target = self.target:long()
         end
         self.target[1] = target
      elseif target:type() == 'torch.CudaTensor' then
         self.target = target
      else
         self.target = target:long()
      end
      self.weights = nil
   end

   self.gradInput:resizeAs(input):zero()

   input.THNN.WeightedClassNLLCriterion_updateGradInput(
      input:cdata(),
      self.target:cdata(),
      self.gradInput:cdata(),
      self.sizeAverage,
      THNN.optionalTensor(self.weights),
      self.total_weight_tensor:cdata()
   )

   return self.gradInput
end
