local MultiMarginCriterion, parent = torch.class('nn.MultiMarginCriterion', 'nn.Criterion')

function MultiMarginCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function MultiMarginCriterion:updateOutput(input, target)
   return input.nn.MultiMarginCriterion_updateOutput(self, input, target)
end

function MultiMarginCriterion:updateGradInput(input, target)
   return input.nn.MultiMarginCriterion_updateGradInput(self, input, target)
end
