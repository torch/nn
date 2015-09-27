local LeakyReLU, parent = torch.class('nn.LeakyReLU','nn.Module')

function LeakyReLU:__init(negval,ip)
   parent.__init(self)
   self.negval = negval or (1/100)
   if  (negval and type(negval) ~= 'number') then
      error('nn.LeakyReLU(value)')
   end
   -- default for inplace is false
   self.inplace = ip or false
   if (ip and type(ip) ~= 'boolean') then
      error('in-place flag must be boolean')
   end
end

function LeakyReLU:updateOutput(input)
   input.nn.LeakyReLU_updateOutput(self, input)
   return self.output
end

function LeakyReLU:updateGradInput(input, gradOutput)
   input.nn.LeakyReLU_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

