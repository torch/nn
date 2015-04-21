local Tanh, parent = torch.class('nn.Tanh', 'nn.Module')

function Tanh:__init(ip)
   parent.__init(self)
   -- default for inplace is false
   self.inplace = ip or false
   if (ip and type(ip) ~= 'boolean') then
      error('in-place flag must be boolean')
   end
end

function Tanh:updateOutput(input)
   self.inplace = self.inplace or false -- backwards compatibility pre inplace
   return input.nn.Tanh_updateOutput(self, input)
end

function Tanh:updateGradInput(input, gradOutput)
   return input.nn.Tanh_updateGradInput(self, input, gradOutput)
end
