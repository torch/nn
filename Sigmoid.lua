local Sigmoid, parent = torch.class('nn.Sigmoid', 'nn.Module')

function Sigmoid:__init(ip)
   parent.__init(self)
   -- default for inplace is false
   self.inplace = ip or false
   if (ip and type(ip) ~= 'boolean') then
      error('in-place flag must be boolean')
   end
end

function Sigmoid:updateOutput(input)
   self.inplace = self.inplace or false -- backwards compatibility pre inplace
   return input.nn.Sigmoid_updateOutput(self, input)
end

function Sigmoid:updateGradInput(input, gradOutput)
   return input.nn.Sigmoid_updateGradInput(self, input, gradOutput)
end
