local CMaxTable, parent = torch.class('nn.CMaxTable', 'nn.Module')

function CMaxTable:__init()
   parent.__init(self)
   self.gradInput = {}
   self.maxIdx = torch.Tensor()
end

function CMaxTable:updateOutput(input)
   self.output:resizeAs(input[1]):copy(input[1])
   self.maxIdx:resizeAs(input[1]):fill(1)
   for i=2,#input do
      local mask = torch.gt(input[i], self.output)
      self.maxIdx:maskedFill(mask, i)
      self.output:maskedCopy(mask, input[i][mask])
   end
   return self.output
end

function CMaxTable:updateGradInput(input, gradOutput)
   for i=1,#input do
      self.gradInput[i] = input[i].new()
      self.gradInput[i]:resizeAs(input[i]):fill(0.0)
      local mask = torch.eq(self.maxIdx, i)
      self.gradInput[i]:maskedCopy(mask, gradOutput[mask])
   end

   for i=#input+1, #self.gradInput do
       self.gradInput[i] = nil
   end

   return self.gradInput
end
