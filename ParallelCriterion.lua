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
   if not self.repeatTarget then
      for i,criterion in ipairs(self.criterions) do
         self.output = self.output + self.weights[i]*criterion:updateOutput(input[i],target[i])
      end
   else
      for i,criterion in ipairs(self.criterions) do
         self.output = self.output + self.weights[i]*criterion:updateOutput(input[i],target)
      end
   end
   return self.output
end

function ParallelCriterion:updateGradInput(input, target)
   if not self.repeatTarget then
      for i,criterion in ipairs(self.criterions) do
         self.gradInput[i] = input[i].new() or self.gradInput[i]
         self.gradInput[i]:resizeAs(input[i]):zero()
         self.gradInput[i]:add(self.weights[i], criterion:updateGradInput(input[i],target[i]))
      end
   else
      for i,criterion in ipairs(self.criterions) do
         self.gradInput[i] = input[i].new() or self.gradInput[i]
         self.gradInput[i]:resizeAs(input[i]):zero()
         self.gradInput[i]:add(self.weights[i], criterion:updateGradInput(input[i],target))
      end
   end
   return self.gradInput
end

function ParallelCriterion:type(type)
   self.gradInput = {}
   return parent.type(self, type)
end
