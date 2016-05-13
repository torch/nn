local SpatialWeightedCrossEntropyCriterion, Criterion = torch.class('nn.SpatialWeightedCrossEntropyCriterion', 'nn.Criterion')

function SpatialWeightedCrossEntropyCriterion:__init()
   Criterion.__init(self)
   self.lsm = nn.LogSoftMax()
   self.nll = nn.WeightedClassNLLCriterion()
end

function SpatialWeightedCrossEntropyCriterion:updateOutput(input, target)
   assert(input:dim() == 4) 
   self.nclass = input:size(2)
   self.prep = nn.Sequential():add(nn.Transpose({3,4},{2,4}))
                              :add(nn.View(-1, self.nclass):setNumInputDims(4))

   self.input = self.prep:forward(input)

   self.lsm:updateOutput(self.input)
   if type(target) == 'table' then -- has weights
      self.target  = target[1]:view(-1)
      self.weights = target[2]:view(-1)
      self.nll:updateOutput(self.lsm.output, {self.target, self.weights})
   else -- w/o weights
      self.target = target:view(-1)
      self.weights = nil
      self.nll:updateOutput(self.lsm.output, self.target)
   end
   self.output = self.nll.output
   return self.output
end

function SpatialWeightedCrossEntropyCriterion:updateGradInput(input, target)
   assert(input:dim() == 4) 
   self.nclass = input:size(2)
   self.prep = nn.Sequential():add(nn.Transpose({3,4},{2,4}))
                              :add(nn.View(-1, self.nclass):setNumInputDims(4))

   local size = input:size()
   --input = input:squeeze()
   self.input = self.prep:forward(input)

   if type(target) == 'table' then -- has weights
      self.target  = target[1]:view(-1)
      self.weights = target[2]:view(-1)
      self.nll:updateGradInput(self.lsm.output, {self.target, self.weights})
   else -- w/o weights
      self.target = target:view(-1)
      self.weights = nil
      self.nll:updateGradInput(self.lsm.output, self.target)
   end
   self.lsm:updateGradInput(self.input, self.nll.gradInput)
   self.gradInput:view(self.lsm.gradInput, size)
   return self.gradInput
end

return nn.SpatialWeightedCrossEntropyCriterion
