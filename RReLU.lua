local RReLU, parent = torch.class('nn.RReLU', 'nn.Module')

function RReLU:__init(l, u, ip)
   parent.__init(self)
   self.lower = l or 1/8
   self.upper = u or 1/3
   assert(self.lower <= self.upper and self.lower >= 0 and self.upper >= 0)
   self.noise = torch.Tensor()
   self.train = true
   self.inplace = ip or false
end

function RReLU:updateOutput(input)
   input.nn.RReLU_updateOutput(self, input)
   return self.output
end

function RReLU:updateGradInput(input, gradOutput)
   input.nn.RReLU_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

function RReLU:__tostring__()
  return string.format('%s (l:%f, u:%f)', torch.type(self), self.lower, self.upper)
end
