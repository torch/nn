local SmoothL1Criterion, parent = torch.class('nn.SmoothL1Criterion', 'nn.Criterion')

function SmoothL1Criterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function SmoothL1Criterion:updateOutput(input, target)
   local delta = target - input
   local loss = delta:abs():apply(
      function(x)
         if x < 1 then
            return 0.5 * x * x
         else
            return x - 0.5
         end 
      end
   ):sum()
   if self.sizeAverage then
      loss = loss / input:nElement()
   end
   return loss
end

function SmoothL1Criterion:updateGradInput(input, target)
   local norm = self.sizeAverage and 1.0 / input:nElement() or 1.0
   self.gradInput:resizeAs(input):copy(input):add(-1, target)
   self.gradInput:apply(
      function(x)
         if math.abs(x) < 1 then
            return norm * x
         elseif x > 0 then
            return norm
         else
            return -norm
         end
      end
   )
   return self.gradInput 
end
