local BatchReshape, parent = torch.class('nn.BatchReshape', 'nn.Module')

--- Reshape, using one dimension to hold the remains
function BatchReshape:__init(...)
   parent.__init(self)
   local arg = {...}

   self.size = torch.LongStorage()
   self.index = arg[1]
   table.remove(arg, 1)

   local n = #arg
   if n == 1 and torch.typename(arg[1]) == 'torch.LongStorage' then
      self.size:resize(#arg[1]):copy(arg[1])
   else
      self.size:resize(n)
      for i=1,n do
         self.size[i] = arg[i]
      end
   end

   self.sizeindex = self.size[self.index]
   self.nelement = 1
   for i=1,#self.size do
      self.nelement = self.nelement * self.size[i]
   end
   
   -- only used for non-contiguous input or gradOutput
   self._input = torch.Tensor()
   self._gradOutput = torch.Tensor()
end

function BatchReshape:updateOutput(input)
   if not input:isContiguous() then
      self._input:resizeAs(input)
      self._input:copy(input)
      input = self._input
   end

   -- Computes the number of elements in the input tensor
   nelement = 1
   for i=1,##input do
      nelement = nelement * (#input)[i]
   end

   assert(nelement % self.nelement == 0, "The number of elements in target shape is not a multiple of the number of elements")

   self.size[self.index] = self.sizeindex * (nelement / self.nelement)
   self.output:view(input, self.size)

   return self.output
end

function BatchReshape:updateGradInput(input, gradOutput)
   if not gradOutput:isContiguous() then
      self._gradOutput:resizeAs(gradOutput)
      self._gradOutput:copy(gradOutput)
      gradOutput = self._gradOutput
   end
   
   self.gradInput:viewAs(gradOutput, input)
   return self.gradInput
end


