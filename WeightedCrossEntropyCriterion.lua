local WeightedCrossEntropyCriterion, Criterion = torch.class('nn.WeightedCrossEntropyCriterion', 'nn.Criterion')

function WeightedCrossEntropyCriterion:__init()
   Criterion.__init(self)
   self.lsm = nn.LogSoftMax()
   self.nll = nn.WeightedClassNLLCriterion()
end

function WeightedCrossEntropyCriterion:updateOutput(input, target)
   self.input = input:squeeze()

   self.lsm:updateOutput(input)
   if type(target) == 'table' then -- has weights
      self.target  = target[1]:squeeze()
      self.weights = target[2]:squeeze()
      self.nll:updateOutput(self.lsm.output, {self.target, self.weights})
   else -- w/o weights
      self.target = target
      self.weights = nil
      self.nll:updateOutput(self.lsm.output, self.target)
   end
   self.output = self.nll.output
   return self.output
end

function WeightedCrossEntropyCriterion:updateGradInput(input, target)
   local size = input:size()
   input = input:squeeze()

   if type(target) == 'table' then -- has weights
      self.target  = target[1]:squeeze()
      self.weights = target[2]:squeeze()
      self.nll:updateGradInput(self.lsm.output, {self.target, self.weights})
   else -- w/o weights
      self.target = target
      self.weights = nil
      self.nll:updateGradInput(self.lsm.output, self.target)
   end
   self.lsm:updateGradInput(input, self.nll.gradInput)
   self.gradInput:view(self.lsm.gradInput, size)
   return self.gradInput
end

return nn.WeightedCrossEntropyCriterion
