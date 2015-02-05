local MultiMarginCriterion, parent = torch.class('nn.MultiMarginCriterion', 'nn.Criterion')

function MultiMarginCriterion:__init(p)
   assert(p == nil or p == 1 or p == 2, 'only p=1 and p=2 supported')
   self.p = p or 1
   parent.__init(self)
   self.sizeAverage = true
end

function MultiMarginCriterion:updateOutput(input, target)
   -- backward compatibility
   self.p = self.p or 1
   return input.nn.MultiMarginCriterion_updateOutput(self, input, target)
end

function MultiMarginCriterion:updateGradInput(input, target)
   return input.nn.MultiMarginCriterion_updateGradInput(self, input, target)
end
