local MultiMarginCriterion, parent = torch.class('nn.MultiMarginCriterion', 'nn.Criterion')

function MultiMarginCriterion:__init(p)
   assert(p == nil or p == 1 or p == 2, 'only p=1 and p=2 supported')
   self.p = p or 1
   parent.__init(self)
   self.sizeAverage = true
end

function MultiMarginCriterion:updateOutput(input, target)
   -- backward compatibility
   local _target = target
   if not torch.isTensor(_target) then
     _target = input.new(1)
     _target[1] = target
   end
   self.p = self.p or 1
   self.output_tensor = self.output_tensor or input.new(1)
   input.THNN.MultiMarginCriterion_updateOutput(
      input:cdata(),
      _target:cdata(),
      self.output_tensor:cdata(),
      self.sizeAverage,
      self.p
   )
   self.output = self.output_tensor[1]
   return self.output
end

function MultiMarginCriterion:updateGradInput(input, target)
   local _target = target
   if not torch.isTensor(_target) then
     _target = input.new(1)
     _target[1] = target
   end
   input.THNN.MultiMarginCriterion_updateGradInput(
      input:cdata(),
      _target:cdata(),
      self.gradInput:cdata(),
      self.sizeAverage,
      self.p
   )
   return self.gradInput
end
