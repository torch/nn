local MarginCriterion, parent = torch.class('nn.MarginCriterion', 'nn.Criterion')

function MarginCriterion:__init(margin)
   parent.__init(self)
   margin=margin or 1   
   self.margin = margin 
   self.buffer = torch.Tensor()
   self.ty = torch.Tensor(1)
end 
 
function MarginCriterion:updateOutput(input, y)
   if not torch.isTensor(y) then self.ty[1]=y; y=self.ty end
   self.buffer:resizeAs(input):fill(self.margin)
   self.buffer:addcmul(-1, input, y)   
   self.buffer[torch.le(self.buffer, 0)] = 0
   self.output = self.buffer:sum()
   return self.output
end

function MarginCriterion:updateGradInput(input, y)
   if not torch.isTensor(y) then self.ty[1]=y; y=self.ty end
   self.gradInput:resizeAs(input):copy(-y)
   self.gradInput[torch.le(self.buffer, 0)] = 0
   return self.gradInput 
end
