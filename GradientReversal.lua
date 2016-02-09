local GradientReversal = torch.class('nn.GradientReversal', 'nn.Module')

function GradientReversal:updateOutput(input)
   self.output:set(input)
   return self.output
end

function GradientReversal:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput)
   self.gradInput:copy(gradOutput)
   self.gradInput:mul(-1)
   return self.gradInput
end
