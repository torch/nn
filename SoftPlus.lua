local SoftPlus, parent = torch.class('nn.SoftPlus', 'nn.Module')


function SoftPlus:__init(beta, ip)
  parent.__init(self)
  self.beta = beta or 1  -- Beta controls sharpness of transfer function
  self.threshold = 20  -- Avoid floating point issues with exp(x), x>20
   -- default for inplace is false
   self.inplace = ip or false
   if (ip and type(ip) ~= 'boolean') then
      error('in-place flag must be boolean')
   end
end

function SoftPlus:updateOutput(input)
   self.inplace = self.inplace or false -- backwards compatibility pre inplace
   -- f(x) = 1/beta * log(1 + exp(beta * x))
   input.nn.SoftPlus_updateOutput(self, input)
   return self.output
end

function SoftPlus:updateGradInput(input, gradOutput)
   -- d/dx[log(1+exp(k*x))/k] = exp(kx) / (exp(kx) + 1)
   -- SINCE
   -- y = (1/k)*log(1+exp(k*x)) --> x = (1/k)*log(exp(k*y)-1)
   -- THEREFORE:
   -- d/dx(f(x)) = (exp(k*y) - 1) / exp(k*y)
   input.nn.SoftPlus_updateGradInput(self, input, gradOutput)
   return self.gradInput
end
