local DistKLDivCriterion, parent = torch.class('nn.DistKLDivCriterion', 'nn.Criterion')

local epsilon = 1e-100

function DistKLDivCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function DistKLDivCriterion:updateOutput(input, target)
   local log = math.log
   if input:dim() == 1 then
      self.output = 0
      for i = 1,input:size(1) do
         local acc = 0
         if target[i] > 0 then
            acc = target[i] * (log(target[i]) - input[i])
         end
         self.output = self.output + acc
      end
   elseif input:dim() == 2 then
      self.output = 0
      for i=1,target:size(1) do
         local tar = target[i]
         local inp = input[i]
         for i = 1,inp:size(1) do
            local acc = 0
            if tar[i] > epsilon then
               acc = tar[i] * (log(tar[i]) - inp[i])
            end
            self.output = self.output + acc
         end
      end
      if self.sizeAverage then
         self.output = self.output / target:size(1)
      end
   else
      error('matrix or vector expected')
   end
   return self.output
end

function DistKLDivCriterion:updateGradInput(input, target)
   local gradInput = self.gradInput
   gradInput:resizeAs(input)

   if input:dim() == 1 then
      for i = 1,input:size(1) do
         gradInput[i] = -target[i]
      end
   else
      for i=1,target:size(1) do
         local tar = target[i]
         for i = 1,tar:size(1) do
            gradInput[i] = -tar[i]
         end
      end
      if self.sizeAverage then
         gradInput:div(target:size(1))
      end
   end

   return self.gradInput
end
