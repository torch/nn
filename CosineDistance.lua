local CosineDistance, parent = torch.class('nn.CosineDistance', 'nn.Module')

function CosineDistance:__init()
   parent.__init(self)
   self.gradInput = {torch.Tensor(), torch.Tensor()}
   self.output=torch.Tensor(1)
end 
 
function CosineDistance:updateOutput(input)
   local input1, input2 = input[1], input[2]
   self.w1 = input1:dot(input2)
   self.w22 = input1:dot(input1)
   self.w2 = math.sqrt(self.w22)
   self.w32 = input2:dot(input2)
   self.w3 = math.sqrt(self.w32)
   self.output[1] = self.w1/self.w2/self.w3
   return self.output
end

function CosineDistance:updateGradInput(input, gradOutput)
   local v1  = input[1]
   local v2  = input[2]
   local gw1 = input[1].new()
   local gw2 = input[2].new()
   gw1:resizeAs(v1) 
   gw2:resizeAs(v1)

   gw1:zero()
   gw1:add(1/(self.w2*self.w3), v2)
   gw1:add(-self.w1/(self.w22*self.w2*self.w3), v1)
   
   gw2:zero()
   gw2:add(1/(self.w2*self.w3), v1)
   gw2:add(-self.w1/(self.w32*self.w2*self.w3), v2)

   gw1:mul(gradOutput[1])
   gw2:mul(gradOutput[1])
   self.gradInput = {gw1, gw2}
   return self.gradInput
end
