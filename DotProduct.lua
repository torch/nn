local DotProduct, parent = torch.class('nn.DotProduct', 'nn.Module')

function DotProduct:__init()
   parent.__init(self)
   self.gradInput = {torch.Tensor(), torch.Tensor()}
   self.output=torch.Tensor(1)
end 
 
function DotProduct:updateOutput(input,y)
   self.output[1] = input[1]:dot(input[2])
   return self.output
end

function DotProduct:updateGradInput(input, gradOutput)
   local v1 = input[1]
   local v2 = input[2]
   local gw1=self.gradInput[1];
   local gw2=self.gradInput[2];
   gw1:resizeAs(v1) 
   gw2:resizeAs(v2)

   gw1:copy( v2)
   gw1:mul(gradOutput[1])
   
   gw2:copy( v1)
   gw2:mul(gradOutput[1])

   return self.gradInput
end
