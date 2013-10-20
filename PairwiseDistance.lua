local PairwiseDistance, parent = torch.class('nn.PairwiseDistance', 'nn.Module')

function PairwiseDistance:__init(p)
   parent.__init(self)

   -- state
   self.gradInput = {torch.Tensor(), torch.Tensor()}
   self.output = torch.Tensor(1)
   self.diff = torch.Tensor()
   self.norm=p
end 
  
function PairwiseDistance:updateOutput(input)
   if input[1]:dim() > 2 then
      error('input must be vector or matrix')
   end
   if input[1]:dim() == 1 then
     -- Reshape the input so it always looks like a batch (avoids multiple
     -- code-paths).  (Clement's good suggestion)
     input[1]:resize(1,input[1]:size(1))
     input[2]:resize(1,input[2]:size(1))
   end
 
   self.diff:resizeAs(input[1])
   self.diff:zero()
   self.diff:add(input[1], -1, input[2])
   self.diff:abs()

   self.output:resize(input[1]:size(1))
   self.output:zero()
   self.output:add(self.diff:pow(self.norm):sum(2))
   self.output:pow(1./self.norm)

   return self.output
end

local function mathsign(x) 
   if x==0 then return  2*torch.random(2)-3; end
   if x>0 then return 1; else return -1; end
end

function PairwiseDistance:updateGradInput(input, gradOutput)
   if input[1]:dim() > 2 then
      error('input must be vector or matrix')
   end

   if input[1]:dim() == 1 then
     -- Reshape the input so it always looks like a batch (avoids multiple
     -- code-paths).  (Clement's good suggestion)
     input[1]:resize(1,input[1]:size(1))
     input[2]:resize(1,input[2]:size(1))
   end

   self.gradInput[1]:resize(input[1]:size()) 
   self.gradInput[2]:resize(input[2]:size()) 
   self.gradInput[1]:copy(input[1])
   self.gradInput[1]:add(-1, input[2]) 
   
   if self.norm==1 then
     self.gradInput[1]:apply(mathsign)
   else
     -- Note: derivative of p-norm:
     -- d/dx_k(||x||_p) = (x_k * abs(x_k)^(p-2)) / (||x||_p)^(p-1)
     if (self.norm > 2) then
        self.gradInput[1]:cmul(self.gradInput[1]:clone():abs():pow(self.norm-2))
     end

     self.outExpand = self.outExpand or self.output.new()
     self.outExpand:resize(self.output:size(1), 1)
     self.outExpand:copy(self.output)
     self.outExpand:add(1.0e-6)  -- Prevent divide by zero errors
     self.outExpand:pow(-(self.norm-1))
     self.gradInput[1]:cmul(self.outExpand:expand(self.gradInput[1]:size(1),
        self.gradInput[1]:size(2)))
   end
   
   self.grad = self.grad or gradOutput.new()
   self.ones = self.ones or gradOutput.new()

   self.grad:resizeAs(input[1]):zero()
   self.ones:resize(input[1]:size(2)):fill(1)

   self.grad:addr(gradOutput, self.ones)
   self.gradInput[1]:cmul(self.grad)
   
   self.gradInput[2]:zero():add(-1, self.gradInput[1])
   return self.gradInput
end
