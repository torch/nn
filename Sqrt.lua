local Sqrt, parent = torch.class('nn.Sqrt','nn.Module')

function Sqrt:__init(b, ip)
   parent.__init(self)
   self.eps = b or 0
   -- default for inplace is false
   self.inplace = ip or false
   if (ip and type(ip) ~= 'boolean') then
      error('in-place flag must be boolean')
   end
end

function Sqrt:updateOutput(input)
   self.inplace = self.inplace or false -- backwards compatibility pre inplace
   self.eps = self.eps or 0
   return input.nn.Sqrt_updateOutput(self,input)
end

function Sqrt:updateGradInput(input, gradOutput)
   return input.nn.Sqrt_updateGradInput(self,input,gradOutput)
end
