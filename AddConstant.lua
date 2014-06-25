local AddConstant, parent = torch.class('nn.AddConstant', 'nn.Module')

function AddConstant:__init(constant_scalar)
  parent.__init(self)
  assert(type(constant_scalar) == 'number', 'input is not scalar!')
  self.constant_scalar = constant_scalar
end

function AddConstant:updateOutput(input)
  self.output:resizeAs(input)
  self.output:copy(input)
  self.output:add(self.constant_scalar)
  return self.output
end 

function AddConstant:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput)
  self.gradInput:copy(gradOutput)
  return self.gradInput
end
