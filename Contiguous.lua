local Contiguous, parent = torch.class('nn.Contiguous', 'nn.Module')

function Contiguous:updateOutput(input)
   if not input:isContiguous() then
      self.output:resizeAs(input):copy(input)
   else
      self.output:set(input)
   end
   return self.output
end

function Contiguous:updateGradInput(input, gradOutput)
   if not gradOutput:isContiguous() then
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   else
      self.gradInput:set(gradOutput)
   end
   return self.gradInput
end
