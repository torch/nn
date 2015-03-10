local CrossEntropyCriterion, Criterion = torch.class('nn.CrossEntropyCriterion', 'nn.Criterion')

function CrossEntropyCriterion:__init(weights)
   Criterion.__init(self)
   self.nll = nn.ClassNLLCriterion(weights)
   self.lsm = nn.LogSoftMax()
end

function CrossEntropyCriterion:updateOutput(input, target)
   input = input:squeeze()
   target = type(target) == 'number' and target or target:squeeze()
   self.lsm:updateOutput(input)
   self.nll:updateOutput(self.lsm.output, target)
   self.output = self.nll.output
   return self.output
end

function CrossEntropyCriterion:updateGradInput(input, target)
   local size = input:size()
   input = input:squeeze()
   target = type(target) == 'number' and target or target:squeeze()
   self.nll:updateGradInput(self.lsm.output, target)
   self.lsm:updateGradInput(input, self.nll.gradInput)
   self.gradInput:view(self.nll.gradInput, size)
   return self.gradInput
end

function CrossEntropyCriterion:type(name)
   Criterion.type(self, name)
   self.lsm:type(name)
   self.nll:type(name)
   return self
end

return nn.CrossEntropyCriterion
