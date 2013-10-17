local MarginRankingCriterion, parent = torch.class('nn.MarginRankingCriterion', 'nn.Criterion')

function MarginRankingCriterion:__init(margin)
   parent.__init(self)
   margin=margin or 1
   self.margin = margin 
   self.gradInput = {torch.Tensor(1), torch.Tensor(1)}
end 
 
function MarginRankingCriterion:updateOutput(input,y)
   if input[1]:size(1) == 1 then
      self.output=math.max(0, -y*(input[1][1]-input[2][1]) + self.margin  ) 
   else
      if type(self.output) == "number" then
         self.output = input[1]:clone()
      end
      self.output = self.output or input[1]:clone()
      self.output:resizeAs(input[1])
      self.output:copy(input[1])

      self.output:add(-1, input[2])
      self.output:mul(-y)
      self.output:add(self.margin)

      self.mask = self.mask or self.output:clone()
      self.mask:resizeAs(self.output)
      self.mask:copy(self.output)

      self.mask:ge(self.output, 0.0)
      self.output:cmul(self.mask)
   end

   return self.output
end

function MarginRankingCriterion:updateGradInput(input, y)
   if input[1]:size(1) == 1 then
      local dist = -y*(input[1][1]-input[2][1]) + self.margin
      if dist < 0 then
         self.gradInput[1][1]=0;
         self.gradInput[2][1]=0;
      else	
         self.gradInput[1][1]=-y
         self.gradInput[2][1]=y
      end
   else
      self.dist = self.dist or input[1].new()
      self.dist = self.dist:resizeAs(input[1]):copy(input[1])
      local dist = self.dist

      dist:add(-1, input[2])
      dist:mul(-y)
      dist:add(self.margin)

      self.mask = self.mask or input[1].new()
      self.mask = self.mask:resizeAs(input[1]):copy(dist)
      local mask = self.mask

      mask:ge(dist, 0)

      self.gradInput[1]:resize(dist:size())
      self.gradInput[2]:resize(dist:size())

      self.gradInput[1]:copy(mask)
      self.gradInput[1]:mul(-y)
      self.gradInput[2]:copy(mask)
      self.gradInput[2]:mul(y)

   end
   return self.gradInput 
end
