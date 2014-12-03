local Transpose, parent = torch.class('nn.Transpose', 'nn.Module')

-- transpose dimensions:
-- n = nn.Transpose({1,4},{1,3})
-- will transpose dims 1 and 4, then 1 and 3...

function Transpose:__init(...)
   parent.__init(self)
   self.permutations = {...}
end

function Transpose:updateOutput(input)
   for _,perm in ipairs(self.permutations) do
      input = input:transpose(perm[1],perm[2])
   end
   self.output:resizeAs(input):copy(input)
   return self.output
end

function Transpose:updateGradInput(input, gradOutput)
   for i = #self.permutations,1,-1 do
      local perm = self.permutations[i]
      gradOutput = gradOutput:transpose(perm[1],perm[2])
   end
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   return self.gradInput
end

