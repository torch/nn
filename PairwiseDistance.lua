local PairwiseDistance, parent = torch.class('nn.PairwiseDistance', 'nn.Module')

function PairwiseDistance:__init(p)
   parent.__init(self)

   -- state
   self.gradInput = {torch.Tensor(), torch.Tensor()}
   self.output = torch.Tensor(1)
   self.norm=p
end 
  
function PairwiseDistance:updateOutput(input)
   if input[1]:dim() == 1 then
      self.output[1]=input[1]:dist(input[2],self.norm)
   elseif input[1]:dim() == 2 then
      self.diff = self.diff or input[1].new()
      self.diff:resizeAs(input[1])

      local diff = self.diff:zero()
      --local diff = torch.add(input[1], -1, input[2])
      diff:add(input[1], -1, input[2])
      diff:abs()

      self.output:resize(input[1]:size(1))
      self.output:zero()
      self.output:add(diff:pow(self.norm):sum(2))
      self.output:pow(1./self.norm)
   else
      error('input must be vector or matrix')
   end

   return self.output
end

local function mathsign(x) 
   if x==0 then return  2*torch.random(2)-3; end
   if x>0 then return 1; else return -1; end
end

function PairwiseDistance:updateGradInput(input, gradOutput)
   self.gradInput[1]:resize(input[1]:size()) 
   self.gradInput[2]:resize(input[2]:size()) 
   self.gradInput[1]:copy(input[1])
   self.gradInput[1]:add(-1, input[2])
   if self.norm==1 then
     self.gradInput[1]:apply(mathsign)
   else
     -- See here for derivative of p-norm:
     -- d/dx_k(||x||_p) = (x_k * abs(x_k)^(p-2)) / (||x||_p)^(p-1)
     -- http://en.wikipedia.org/wiki/Norm_(mathematics)
     self.gradInput[1]:cmul(torch.abs(self.gradInput[1]):pow(self.norm-2))
     if input[1]:dim() == 1 then
        -- Avoid the expand for dimension 1
        self.gradInput[1]:mul(math.pow(self.output[1],-(self.norm-1)))
     elseif input[1]:dim() == 2 then
        -- This is a little messy...  But it does work
        self.outExpand = self.outExpand or self.output.new()
        self.outExpand:resize(self.output:size(1), 1)
        self.outExpand:copy(self.output)
        self.outExpand:pow(-(self.norm-1))
        self.gradInput[1]:cmul(self.outExpand:expand(self.gradInput[1]:size(1),
           self.gradInput[1]:size(2)))
     else
        error('input must be vector or matrix')
     end
   end
   if input[1]:dim() == 1 then
      self.gradInput[1]:mul(gradOutput[1])
   elseif input[1]:dim() == 2 then
      self.grad = self.grad or gradOutput.new()
      self.ones = self.ones or gradOutput.new()

      self.grad:resizeAs(input[1]):zero()
      self.ones:resize(input[1]:size(2)):fill(1)

      self.grad:addr(gradOutput, self.ones)
      self.gradInput[1]:cmul(self.grad)
   else
      error('input must be vector or matrix')
   end
   self.gradInput[2]:zero():add(-1, self.gradInput[1])
   return self.gradInput
end
