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
      self.w2  = input1.new()
      self.w32 = input1.new()
      self.w3  = input1.new()
   end
 
   self.buffer:cmul(input1,input2)
   self.w1:sum(self.buffer,2)

   self.buffer:cmul(input1,input1)
   self.w22:sum(self.buffer,2)
   self.w2:sqrt(self.w22)

   self.buffer:cmul(input2,input2)
   self.w32:sum(self.buffer,2)
   self.w3:sqrt(self.w32)

   self.output:cdiv(self.w1,self.w2):cdiv(self.w3)
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
   gw1:resizeAs(v1):zero() 
   gw2:resizeAs(v1):zero()

   self.buffer:cmul(self.w2,self.w3)
   self.buffer = self.buffer:expandAs(v1)

   gw1:cdiv(v2,self.buffer)
   gw2:cdiv(v1,self.buffer)

   self.buffer:cdiv(self.w1,self.w22):cdiv(self.w2):cdiv(self.w3)
   self.buffer = self.buffer:expandAs(v1)
   gw1:addcmul(-1,self.buffer,v1)

   self.buffer:cdiv(self.w1,self.w32):cdiv(self.w2):cdiv(self.w3)
   self.buffer = self.buffer:expandAs(v1)
   gw2:addcmul(-1,self.buffer,v2)

   local go = gradOutput:view(-1,1):expandAs(v1)
   gw1:cmul(go)
   gw2:cmul(go)

   if not_batch then
      self.gradInput[1] = gw1:select(1,1)
      self.gradInput[2] = gw2:select(1,1)
   end

   return self.gradInput
end
