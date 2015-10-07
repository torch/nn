local HardTanh, parent = torch.class('nn.HardTanh', 'nn.Module')

function HardTanh:__init(min_value, max_value)
   parent.__init(self)
   self.min_val = min_value or -1
   self.max_val = max_value or 1
   assert(self.max_val>self.min_val, 'max_value must be larger than min_value')
end

function HardTanh:updateOutput(input)
   self.min_val = self.min_val or -1
   self.max_val = self.max_val or 1
   return input.nn.HardTanh_updateOutput(self, input)
end

function HardTanh:updateGradInput(input, gradOutput)
   return input.nn.HardTanh_updateGradInput(self, input, gradOutput)
end
