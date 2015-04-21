local Exp, parent = torch.class('nn.Exp', 'nn.Module')

function Exp:__init(ip)
   parent.__init(self)
   -- default for inplace is false
   self.inplace = ip or false
   if (ip and type(ip) ~= 'boolean') then
      error('in-place flag must be boolean')
   end
end

function Exp:updateOutput(input)
  if self.inplace then
     input:exp()
     self.output = input
  else
     self.output:exp(input)
  end
  return self.output
end

function Exp:updateGradInput(input, gradOutput)
  if self.inplace then
     gradOutput:cmul(self.output)
     self.gradInput = gradOutput
  else
     self.gradInput:cmul(self.output, gradOutput)
  end
  return self.gradInput
end
