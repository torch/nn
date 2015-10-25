local LeakyReLU, parent = torch.class('nn.LeakyReLU','nn.Module')

function LeakyReLU:__init(negval,ip)
   parent.__init(self)
   if type(negval) == 'boolean' then
	 local ip = negval
	 self.negval = 1/100
   else
	 self.negval = negval or (1/100)
   end
   -- default for inplace is false
   self.inplace = ip or false
   if self.negval < 0 then
      self.inplace = false
   end
end

function LeakyReLU:__tostring__()
  return torch.type(self) .. string.format('(%g)', self.negval)
end


function LeakyReLU:updateOutput(input)
   input.nn.LeakyReLU_updateOutput(self, input)
   return self.output
end

function LeakyReLU:updateGradInput(input, gradOutput)
   input.nn.LeakyReLU_updateGradInput(self, input, gradOutput)
   return self.gradInput
end


