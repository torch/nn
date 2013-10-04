local MultiLabelMarginCriterion, parent = torch.class('nn.MultiLabelMarginCriterion', 'nn.Criterion')

function MultiLabelMarginCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function MultiLabelMarginCriterion:updateOutput(input, target)
   return input.nn.MultiLabelMarginCriterion_updateOutput(self, input, target)
end

function MultiLabelMarginCriterion:updateGradInput(input, target)
   return input.nn.MultiLabelMarginCriterion_updateGradInput(self, input, target)
end
