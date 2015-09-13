local SmoothL1Criterion, parent = torch.class('nn.SmoothL1Criterion', 'nn.Criterion')

function SmoothL1Criterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function SmoothL1Criterion:updateOutput(input, target)
   return input.nn.SmoothL1Criterion_updateOutput(self, input, target)
end

function SmoothL1Criterion:updateGradInput(input, target)
   return input.nn.SmoothL1Criterion_updateGradInput(self, input, target)
end
