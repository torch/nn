local BCECriterion, parent = torch.class('nn.BCECriterion', 'nn.Criterion')

local eps = 1e-12

function BCECriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function BCECriterion:updateOutput(input, target)
   -- log(input) * target + log(1 - input) * (1 - target)

   self.term1 = self.term1 or input.new()
   self.term2 = self.term2 or input.new()
   self.term3 = self.term3 or input.new()

   self.term1:resizeAs(input)
   self.term2:resizeAs(input)
   self.term3:resizeAs(input)

   self.term1:fill(1):add(-1,target)
   self.term2:fill(1):add(-1,input):add(eps):log():cmul(self.term1)

   self.term3:copy(input):add(eps):log():cmul(target)
   self.term3:add(self.term2)

   if self.sizeAverage then
      self.term3:div(target:nElement())
   end

   self.output = - self.term3:sum()

   return self.output
end

function BCECriterion:updateGradInput(input, target)
   -- target / input - (1 - target) / (1 - input)

   self.term1 = self.term1 or input.new()
   self.term2 = self.term2 or input.new()
   self.term3 = self.term3 or input.new()

   self.term1:resizeAs(input)
   self.term2:resizeAs(input)
   self.term3:resizeAs(input)

   self.term1:fill(1):add(-1,target)
   self.term2:fill(1):add(-1,input)

   self.term2:add(eps)
   self.term1:cdiv(self.term2)

   self.term3:copy(input):add(eps)

   self.gradInput:resizeAs(input)
   self.gradInput:copy(target):cdiv(self.term3)

   self.gradInput:add(-1,self.term1)

   if self.sizeAverage then
      self.gradInput:div(target:nElement())
   end

   self.gradInput:mul(-1)

   return self.gradInput
end
