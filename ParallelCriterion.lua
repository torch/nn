local ParallelCriterion, parent = torch.class('nn.ParallelCriterion', 'nn.Criterion')

function ParallelCriterion:__init(repeatTarget)
   parent.__init(self)
   self.criterions = {}
   self.weights = {}
   self.gradInput = {}
   self.repeatTarget = repeatTarget
end

function ParallelCriterion:add(criterion, weight)
   weight = weight or 1
   table.insert(self.criterions, criterion)
   table.insert(self.weights, weight)
   return self
end

function ParallelCriterion:updateOutput(input, target)
   self.output = 0
   for i,criterion in ipairs(self.criterions) do
      local target = self.repeatTarget and target or target[i]
      self.output = self.output + self.weights[i]*criterion:updateOutput(input[i],target)
   end
   return self.output
end

function ParallelCriterion:updateGradInput(input, target)
   for i,criterion in ipairs(self.criterions) do
      local target = self.repeatTarget and target or target[i]
      self.gradInput[i] = self.gradInput[i] or input[i].new()
      self.gradInput[i]:resizeAs(input[i]):zero()
      self.gradInput[i]:add(self.weights[i], criterion:updateGradInput(input[i],target))
   end
   return self.gradInput
end

function ParallelCriterion:type(type)
   self.gradInput = {}
   return parent.type(self, type)
end
