local Power, parent = torch.class('nn.Power','nn.Module')

function Power:__init(p)
   parent.__init(self)
   self.pow = p
   if not p then
      error('nn.Power(power)')
   end
end

function Power:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   self.output:pow(self.pow)
   return self.output
end

function Power:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):copy(gradOutput)
   self.gradInput:cmul(torch.pow(input,self.pow-1)):mul(self.pow)
   return self.gradInput
end