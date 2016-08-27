local THNN = require 'nn.THNN'
local WeightedSmoothL1Criterion, parent = torch.class('nn.WeightedSmoothL1Criterion', 'nn.Criterion')

function WeightedSmoothL1Criterion:__init(sizeAverage)
   parent.__init(self)
   if sizeAverage ~= nil then
     self.sizeAverage = sizeAverage
   else
     self.sizeAverage = true
   end

   self.total_weight_tensor = torch.ones(1)
end

function WeightedSmoothL1Criterion:updateOutput(input, target)
   if type(target) == 'table' then -- has weights
      self.target  = target[1]
      self.weights = target[2]
   else -- w/o weights
      self.target = target
      self.weights = nil
   end

   self.output_tensor = self.output_tensor or input.new(1)
   input.THNN.WeightedSmoothL1Criterion_updateOutput(
      input:cdata(),
      self.target:cdata(),
      self.output_tensor:cdata(),
      self.sizeAverage, 
      THNN.optionalTensor(self.weights),
      self.total_weight_tensor:cdata()
   )
   self.output = self.output_tensor[1]
   return self.output
end

function WeightedSmoothL1Criterion:updateGradInput(input, target)
   if type(target) == 'table' then -- has weights
      self.target  = target[1]
      self.weights = target[2]
   else -- w/o weights
      self.target = target
      self.weights = nil
   end

   self.gradInput:resizeAs(input):zero()

   input.THNN.WeightedSmoothL1Criterion_updateGradInput(
      input:cdata(),
      self.target:cdata(),
      self.gradInput:cdata(),
      self.sizeAverage, 
      THNN.optionalTensor(self.weights),
      self.total_weight_tensor:cdata()
   )
   return self.gradInput
end
