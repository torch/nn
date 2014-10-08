local MarginCriterion, parent = torch.class('nn.MarginCriterion', 'nn.Criterion')

function MarginCriterion:__init(margin)
   parent.__init(self)
   self.sizeAverage = true
   self.margin = margin or 1
end

function MarginCriterion:updateOutput(input, target)
   return input.nn.MarginCriterion_updateOutput(self, input, target)
end

function MarginCriterion:updateGradInput(input, target)
   input.nn.MarginCriterion_updateGradInput(self, input, target)
   return self.gradInput
end
