local AddConstant, parent = torch.class('nn.AddConstant', 'nn.Module')

function AddConstant:__init(constant_scalar,ip)
  parent.__init(self)
  assert(type(constant_scalar) == 'number', 'input is not scalar!')
  self.constant_scalar = constant_scalar
  
  -- default for inplace is false
   self.inplace = ip or false
   if (ip and type(ip) ~= 'boolean') then
      error('in-place flag must be boolean')
   end
end

function AddConstant:updateOutput(input)
  if self.inplace then
    input:add(self.constant_scalar)
    self.output:set(input)
  else
    self.output:resizeAs(input)
    self.output:copy(input)
    self.output:add(self.constant_scalar)
  end
  return self.output
end 

function AddConstant:updateGradInput(input, gradOutput)
  if self.inplace then
    self.gradInput:set(gradOutput)
    -- restore previous input value
    input:add(-self.constant_scalar)
  else
    self.gradInput:resizeAs(gradOutput)
    self.gradInput:copy(gradOutput)
  end
  return self.gradInput
end
