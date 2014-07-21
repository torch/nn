local DistKLDivCriterion, parent = torch.class('nn.DistKLDivCriterion', 'nn.Criterion')

function DistKLDivCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function DistKLDivCriterion:updateOutput(input, target)
   return input.nn.DistKLDivCriterion_updateOutput(self, input, target)  
end

function DistKLDivCriterion:updateGradInput(input, target)
   return input.nn.DistKLDivCriterion_updateGradInput(self, input, target)
end
