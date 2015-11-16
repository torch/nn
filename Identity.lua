local Identity, _ = torch.class('nn.Identity', 'nn.Module')

function Identity:updateOutput(input)
   self.output:set(input:storage(), input:storageOffset(), input:size(), input:stride())
   return self.output
end


function Identity:updateGradInput(input, gradOutput)
   self.gradInput:set(gradOutput:storage(), gradOutput:storageOffset(), gradOutput:size(), gradOutput:stride())
   return self.gradInput
end
