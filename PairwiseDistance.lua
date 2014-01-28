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
   if input[1]:dim() == 1 then
      self.output[1]=input[1]:dist(input[2],self.norm)
   elseif input[1]:dim() == 2 then
      self.diff = self.diff or input[1].new()
      self.diff:resizeAs(input[1])

      local diff = self.diff:zero()
      diff:add(input[1], -1, input[2])
      diff:abs()

      self.output:resize(input[1]:size(1))
      self.output:zero()
      self.output:add(diff:pow(self.norm):sum(2))
      self.output:pow(1./self.norm)
   else
      error('input must be vector or matrix')
   end
   if input[1]:dim() > 2 then
      error('input must be vector or matrix')
   end  
 
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

     if (input[1]:dim() > 1) then
        self.outExpand = self.outExpand or self.output.new()
        self.outExpand:resize(self.output:size(1), 1)
        self.outExpand:copy(self.output)
        self.outExpand:add(1.0e-6)  -- Prevent divide by zero errors
        self.outExpand:pow(-(self.norm-1))
        self.gradInput[1]:cmul(self.outExpand:expand(self.gradInput[1]:size(1),
           self.gradInput[1]:size(2)))
     else
        self.gradInput[1]:mul(math.pow(self.output[1] + 1e-6, -(self.norm-1)))
     end
   end
   if input[1]:dim() == 1 then
      self.gradInput[1]:mul(gradOutput[1])
   else
      self.grad = self.grad or gradOutput.new()
      self.ones = self.ones or gradOutput.new()

      self.grad:resizeAs(input[1]):zero()
      self.ones:resize(input[1]:size(2)):fill(1)

      self.grad:addr(gradOutput, self.ones)
      self.gradInput[1]:cmul(self.grad)
   end
   self.gradInput[2]:zero():add(-1, self.gradInput[1])
   return self.gradInput
end

-- save away Module:type(type) for later use.
PairwiseDistance._parent_type = parent.type

-- Fix the bug where tmp = nn.PairwiseDistance:cuda() fails to convert table
-- contents.  We could, and probably should, change Module.lua to loop over
-- and convert all the table elements in a module, but that might have 
-- repercussions, so this is a safer solution.
function PairwiseDistance:type(type)
   self:_parent_type(type)  -- Call the parent (Module) type function
   -- Now convert the left over table elements
   self.gradInput[1] = self.gradInput[1]:type(type)
   self.gradInput[2] = self.gradInput[2]:type(type)
   return self
end

