local Sqrt, parent = torch.class('nn.Sqrt','nn.Module')

function Sqrt:__init(b)
   parent.__init(self)
   self.eps = b or 0
end

function Sqrt:updateOutput(input)
   self.eps = self.eps or 0
   return input.nn.Sqrt_updateOutput(self,input)
end

function Sqrt:updateGradInput(input, gradOutput)
   return input.nn.Sqrt_updateGradInput(self,input,gradOutput)
end
