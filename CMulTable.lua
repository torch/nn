
local CMulTable, parent = torch.class('nn.CMulTable', 'nn.Module')

function CMulTable:__init()
   parent.__init(self)
   self.gradInput = {}
end

function CMulTable:updateOutput(input)
   self.output:resizeAs(input[1]):copy(input[1])
   for i=2,#input do
      self.output:cmul(input[i])
   end
   return self.output
end

function CMulTable:updateGradInput(input, gradOutput)
   local tout = torch.Tensor():resizeAs(self.output)
   for i=1,#input do
      self.gradInput[i] = self.gradInput[i] or torch.Tensor()
      self.gradInput[i]:resizeAs(input[i]):copy(gradOutput)
      tout:copy(self.output):cdiv(input[i])
      self.gradInput[i]:cmul(tout)
   end
   return self.gradInput
end
