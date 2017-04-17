local HingeEmbeddingCriterionBatch, parent = torch.class('nn.HingeEmbeddingCriterionBatch', 'nn.Criterion')

function HingeEmbeddingCriterionBatch:__init(margin)
   parent:__init(self)
   margin = margin or 1
   self.margin = margin
   self.margin_input = torch.Tensor()
end

function HingeEmbeddingCriterionBatch:updateOutput(input,y)
   if y:eq(1):sum()+y:eq(-1):sum()~=y:nElement() then
      error('The target values should be either 1 or -1.')
   end
   self.output = torch.Tensor():typeAs(input):resizeAs(input):fill(0)
   self.output:add(y:eq(1):typeAs(input):cmul(input))
   self.margin_input = torch.mul(input,-1):add(self.margin)
   self.margin_input:cmul(self.margin_input:gt(0):typeAs(self.margin_input))
   self.output:add(y:eq(-1):typeAs(input):cmul(self.margin_input))
   return self.output
end

function HingeEmbeddingCriterionBatch:updateGradInput(input, y)
   self.gradInput = torch.Tensor():typeAs(input):resizeAs(input):fill(0)
   self.gradInput:add(y:eq(1):typeAs(input))
   self.gradInput:add(y:eq(-1):cmul(self.margin_input:gt(0)):typeAs(input):mul(-1))
   return self.gradInput
end
