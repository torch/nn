local CosineDistance, parent = torch.class('nn.CosineDistance', 'nn.Module')

function CosineDistance:__init()
   parent.__init(self)
   self.gradInput = {torch.Tensor(), torch.Tensor()}
end 
 
function CosineDistance:updateOutput(input)
   local input1, input2 = input[1], input[2]

   if input1:dim() == 1 then
      input1 = input1:view(1,-1)
      input2 = input2:view(1,-1)
   end

   if not self.buffer then
      self.buffer = input1.new()
      self.w1  = input1.new()
      self.w22 = input1.new()
      self.w  = input1.new()
      self.w32 = input1.new()
      self.ones = input1.new()
   end

   self.buffer:cmul(input1,input2)
   self.w1:sum(self.buffer,2)

   local epsilon = 1e-12
   self.buffer:cmul(input1,input1)
   self.w22:sum(self.buffer,2):add(epsilon)
   self.ones:resizeAs(self.w22):fill(1)
   self.w22:cdiv(self.ones, self.w22)
   self.w:resizeAs(self.w22):copy(self.w22)

   self.buffer:cmul(input2,input2)
   self.w32:sum(self.buffer,2):add(epsilon)
   self.w32:cdiv(self.ones, self.w32)
   self.w:cmul(self.w32)
   self.w:sqrt()

   self.output:cmul(self.w1,self.w)
   self.output = self.output:select(2,1)

   return self.output
end

function CosineDistance:updateGradInput(input, gradOutput)
   local v1  = input[1]
   local v2  = input[2]
   local not_batch = false

   if v1:dim() == 1 then
      v1 = v1:view(1,-1)
      v2 = v2:view(1,-1)
      not_batch = true
   end

   local gw1 = self.gradInput[1]
   local gw2 = self.gradInput[2]
   gw1:resizeAs(v1):copy(v2)
   gw2:resizeAs(v1):copy(v1)

   self.w = self.w:expandAs(v1)
   self.buffer:cmul(self.w1,self.w22)
   self.buffer = self.buffer:expandAs(v1)
   gw1:addcmul(-1,self.buffer,v1)
   gw1:cmul(self.w)

   self.buffer:cmul(self.w1,self.w32)
   self.buffer = self.buffer:expandAs(v1)
   gw2:addcmul(-1,self.buffer,v2)
   gw2:cmul(self.w)

   local go = gradOutput:view(-1,1):expandAs(v1)
   gw1:cmul(go)
   gw2:cmul(go)

   if not_batch then
      self.gradInput[1] = gw1:select(1,1)
      self.gradInput[2] = gw2:select(1,1)
   end

   -- fix for torch bug 
   -- https://github.com/torch/torch7/issues/289
   self.buffer:resize()

   return self.gradInput
end
