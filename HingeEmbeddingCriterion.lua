local HingeEmbeddingCriterion, parent = torch.class('nn.HingeEmbeddingCriterion', 'nn.Criterion')

function HingeEmbeddingCriterion:__init(margin)
   parent.__init(self)
   margin=margin or 1 
   self.margin = margin 
   self.gradInput = torch.Tensor(1)
end 
 
function HingeEmbeddingCriterion:updateOutput(input,y)
   self.output=input[1]
   if y==-1 then
	 self.output = math.max(0,self.margin - self.output);
   end
   return self.output
end

function HingeEmbeddingCriterion:updateGradInput(input, y)
  self.gradInput[1]=y
  local dist = input[1]
  if y == -1 and  dist > self.margin then
     self.gradInput[1]=0;
  end
  return self.gradInput 
end
