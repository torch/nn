local AbsCriterion, parent = torch.class('nn.AbsCriterion', 'nn.Criterion')

function AbsCriterion:__init(sizeAverage)
   parent.__init(self)
   if sizeAverage ~= nil then
     self.sizeAverage = sizeAverage
   else
     self.sizeAverage = true
   end
end

function AbsCriterion:updateOutput(input, target)
   return input.nn.AbsCriterion_updateOutput(self, input, target)
end

function AbsCriterion:updateGradInput(input, target)
   return input.nn.AbsCriterion_updateGradInput(self, input, target)
end
