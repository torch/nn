local SoftShrink, parent = torch.class('nn.SoftShrink', 'nn.Module')

function SoftShrink:__init(lam)
   parent.__init(self)
   self.lambda = lam or 0.5
end

function SoftShrink:updateOutput(input)
   input.nn.SoftShrink_updateOutput(self, input)
   return self.output
end

function SoftShrink:updateGradInput(input, gradOutput)
   input.nn.SoftShrink_updateGradInput(self, input, gradOutput)
   return self.gradInput
end
