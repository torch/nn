local MulConstant, parent = torch.class('nn.MulConstant', 'nn.Module')

function MulConstant:__init(constant_scalar)
  parent.__init(self)
  assert(type(constant_scalar) == 'number', 'input is not scalar!')
  self.constant_scalar = constant_scalar
end

function MulConstant:updateOutput(input)
  self.output:resizeAs(input)
  self.output:copy(input)
  self.output:mul(self.constant_scalar)
  return self.output
end 

function MulConstant:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput)
  self.gradInput:copy(gradOutput)
  self.gradInput:mul(self.constant_scalar)
  return self.gradInput
end
