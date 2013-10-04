local CosineEmbeddingCriterion, parent = torch.class('nn.CosineEmbeddingCriterion', 'nn.Criterion')

function CosineEmbeddingCriterion:__init(margin)
   parent.__init(self)
   margin=margin or 0
   self.margin = margin 
   self.gradInput = {torch.Tensor(), torch.Tensor()}
end 
 
function CosineEmbeddingCriterion:updateOutput(input,y)
   local input1, input2 = input[1], input[2]
   self.w1 = input1:dot(input2)
   self.w22 = input1:dot(input1)
   self.w2 = math.sqrt(self.w22)
   self.w32 = input2:dot(input2)
   self.w3 = math.sqrt(self.w32)
   self.output = self.w1/self.w2/self.w3
   if y==-1 then
      self.output = math.max(0, self.output - self.margin);
   else
      self.output = 1 - self.output
   end
   return self.output
end

local function mathsign(t)
   if t>0 then return 1; end
   if t<0 then return -1; end
   return 2*torch.random(2)-3;
end

function CosineEmbeddingCriterion:updateGradInput(input, y)
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

   if y == 1 then
      gw1 = -gw1
      gw2 = -gw2
   end
   self.gradInput = {gw1, gw2}
   return self.gradInput
end
