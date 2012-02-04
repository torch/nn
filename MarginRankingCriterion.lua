local MarginRankingCriterion, parent = torch.class('nn.MarginRankingCriterion', 'nn.Criterion')

function MarginRankingCriterion:__init(margin)
   parent.__init(self)
   margin=margin or 1
   self.margin = margin 
   self.gradInput = {torch.Tensor(1), torch.Tensor(1)}
end 
 
function MarginRankingCriterion:updateOutput(input,y)
   self.output=math.max(0, -y*(input[1][1]-input[2][1]) + self.margin  ) 
   return self.output
end

function MarginRankingCriterion:updateGradInput(input, y)
  local dist = -y*(input[1][1]-input[2][1]) + self.margin
  if dist < 0 then
     self.gradInput[1][1]=0;
     self.gradInput[2][1]=0;
  else	
     self.gradInput[1][1]=-y
     self.gradInput[2][1]=y
  end
  return self.gradInput 
end
