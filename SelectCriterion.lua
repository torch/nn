local SelectCriterion, parent = torch.class('nn.SelectCriterion', 'nn.Module')

function SelectCriterion:__init(index)
  parent.__init(self)
  self.index = index
  self.gradInput = {}
end

function SelectCriterion:updateOutput(input)
   local criterions, preds, targets = input[1], input[2], input[3]
   assert(math.abs(self.index) <= #criterions, "arg 1 table idx out of range")
   if self.index < 0 then
      local index = #criterions + self.index + 1
      self.output = criterions[index]:updateOutput(preds, targets)
   else
      self.output = criterions[self.index]:updateOutput(preds, targets)
   end

   return self.output
end

local function zeroCriterionCopy(t1, t2)
end

function SelectCriterion:updateGradInput(input)
   local criterions, preds, targets = input[1], input[2], input[3]
   local index = 0 

   if self.index < 0 then
      index = #criterions + self.index + 1
   else
      index = self.index
   end
  
   local gradInput = criterions[index]:backward(preds, targets)
   self.gradInput[index] = gradInput 

   for i=1, #criterions do
       if i ~= index then
         self.gradInput[i] = torch.zeros(preds:size())
       end
   end

   return self.gradInput
end

function SelectCriterion:type(type, tensorCache)
   self.gradInput = {}
   self.output = {}
   return parent.type(self, type, tensorCache)
end
