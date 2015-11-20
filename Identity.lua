local Identity, parent = torch.class('nn.Identity', 'nn.Module')

function Identity:__init()
   parent.__init(self)
   self.tensorOutput = torch.Tensor{}
   self.output = self.tensorOutput
   self.tensorGradInput = torch.Tensor{}
   self.gradInput = self.tensorGradInput
end

function Identity:updateOutput(input)
   if torch.isTensor(input) then
      self.tensorOutput:set(input)
      self.output = self.tensorOutput
   else
      self.output = input
   end
   return self.output
end


function Identity:updateGradInput(input, gradOutput)
   if torch.isTensor(gradOutput) then
      self.tensorGradInput:set(gradOutput)
      self.gradInput = self.tensorGradInput
   else
      self.gradInput = gradOutput
   end
   return self.gradInput
end
