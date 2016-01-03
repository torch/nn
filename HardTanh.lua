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
   input.THNN.HardTanh_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.min_val,
      self.max_val
   )
   return self.output
end

function HardTanh:updateGradInput(input, gradOutput)
   input.THNN.HardTanh_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.min_val,
      self.max_val
   )
   return self.gradInput
end
