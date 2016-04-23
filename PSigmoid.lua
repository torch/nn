local PSigmoid, parent = torch.class('nn.PSigmoid','nn.Module')

function PSigmoid:__init(k)
   parent.__init(self)
   self.k = k or 1
end

function PSigmoid:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   self.output:mul(-self.k):exp():add(1):cinv()
   return self.output
end

function PSigmoid:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   self.gradInput:fill(1):csub(self.output):cmul(self.output):mul(self.k)
   return self.gradInput
end
