local SmoothL1Criterion, parent = torch.class('nn.SmoothL1Criterion', 'nn.Criterion')

function SmoothL1Criterion:__init(sizeAverage)
   parent.__init(self)
   if sizeAverage ~= nil then
     self.sizeAverage = sizeAverage
   else
     self.sizeAverage = true
   end
end

function SmoothL1Criterion:updateOutput(input, target)
   return input.nn.SmoothL1Criterion_updateOutput(self, input, target)
end

function SmoothL1Criterion:updateGradInput(input, target)
   return input.nn.SmoothL1Criterion_updateGradInput(self, input, target)
end
