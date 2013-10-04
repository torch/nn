local MarginCriterion, parent = torch.class('nn.MarginCriterion', 'nn.Criterion')

function MarginCriterion:__init(margin)
   parent.__init(self)
   margin=margin or 1   
   self.margin = margin 
   self.gradInput = torch.Tensor(1)
end 
 
function MarginCriterion:updateOutput(input,y)
   self.output=math.max(0, self.margin- y* input[1])
   return self.output
end

function MarginCriterion:updateGradInput(input, y)
  if (y*input[1])<self.margin then
     self.gradInput[1]=-y		
  else
     self.gradInput[1]=0;
  end
  return self.gradInput 
end
