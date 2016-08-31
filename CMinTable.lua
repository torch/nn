local CMinTable, parent = torch.class('nn.CMinTable', 'nn.Module')

function CMinTable:__init()
   parent.__init(self)
   self.gradInput = {}
   self.minIdx = torch.Tensor()
end

function CMinTable:updateOutput(input)
   self.output:resizeAs(input[1]):copy(input[1])
   self.minIdx:resizeAs(input[1]):fill(1)
   for i=2,#input do
      local mask = torch.lt(input[i], self.output)
      self.minIdx:maskedFill(mask, i)
      self.output:maskedCopy(mask, input[i][mask])
   end
   return self.output
end

function CMinTable:updateGradInput(input, gradOutput)
   for i=1,#input do
      self.gradInput[i] = torch.Tensor()
      self.gradInput[i]:resizeAs(input[i]):fill(0.0)
      local mask = torch.eq(self.minIdx, i)
      self.gradInput[i]:maskedCopy(mask, gradOutput[mask])
   end

   for i=#input+1, #self.gradInput do
       self.gradInput[i] = nil
   end

   return self.gradInput
end
