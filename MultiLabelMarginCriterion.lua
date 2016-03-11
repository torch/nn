local MultiLabelMarginCriterion, parent = torch.class('nn.MultiLabelMarginCriterion', 'nn.Criterion')

function MultiLabelMarginCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function MultiLabelMarginCriterion:updateOutput(input, target)
   self.output_tensor = self.output_tensor or input.new(1)
   input.THNN.MultiLabelMarginCriterion_updateOutput(
      input:cdata(),
      target:cdata(),
      self.output_tensor:cdata(),
      self.sizeAverage
   )
   self.output = self.output_tensor[1]
   return self.output
end

function MultiLabelMarginCriterion:updateGradInput(input, target)
   input.THNN.MultiLabelMarginCriterion_updateGradInput(
      input:cdata(),
      target:cdata(),
      self.gradInput:cdata(),
      self.sizeAverage
   )
   return self.gradInput
end
