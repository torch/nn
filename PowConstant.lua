local PowConstant, parent = torch.class('nn.PowConstant', 'nn.Module')

function PowConstant:__init(constant_scalar,ip)
  parent.__init(self)
  assert(type(constant_scalar) == 'number', 'input is not scalar!')
  assert(constant_scalar ~= 0, 'zero is not permitted')
  self.constant_scalar = constant_scalar

  -- default for inplace is false
   self.inplace = ip or false
   if (ip and type(ip) ~= 'boolean') then
      error('in-place flag must be boolean')
   end
end

function PowConstant:updateOutput(input)
  if self.inplace then
    input:pow(self.constant_scalar)
    self.output = input
  else
    self.output:resizeAs(input)
    self.output:copy(input)
    self.output:pow(self.constant_scalar)
  end
  return self.output
end

function PowConstant:updateGradInput(input, gradOutput)
  if self.gradInput then
    if self.inplace then
      input:pow((self.constant_scalar-1)/self.constant_scalar)
      gradOutput:mul(input)
      gradOutput:mul(self.constant_scalar)
      self.gradInput = gradOutput
      -- restore previous input value
      input:pow(1/(self.constant_scalar-1))
    else
      self.gradInput:resizeAs(gradOutput)
      self.gradInput:pow(input, self.constant_scalar-1)
      self.gradInput:mul(self.constant_scalar)
      self.gradInput:mul(gradOutput)
    end
    return self.gradInput
  end
end
