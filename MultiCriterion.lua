local MultiCriterion, parent = torch.class('nn.MultiCriterion', 'nn.Criterion')

function MultiCriterion:__init()
   parent.__init(self)
   self.criterions = {}
   self.weights = torch.DoubleStorage()
end

function MultiCriterion:add(criterion, weight)
   weight = weight or 1
   table.insert(self.criterions, criterion)
   self.weights:resize(#self.criterions, true)
   self.weights[#self.criterions] = weight
   return self
end

function MultiCriterion:updateOutput(input, target)
   self.output = 0
   for i=1,#self.criterions do
      self.output = self.output + self.weights[i]*self.criterions[i]:updateOutput(input, target)
   end
   return self.output
end

function MultiCriterion:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()
   for i=1,#self.criterions do
      self.gradInput:add(self.weights[i], self.criterions[i]:updateGradInput(input, target))
   end
   return self.gradInput
end
