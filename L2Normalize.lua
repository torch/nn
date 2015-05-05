
--[[ 
   This layer expects an [n x d] Tensor and normalizes each
   row to have unit L2 norm.
]]--
local L2Normalize, parent = torch.class('nn.L2Normalize', 'nn.Module')
function L2Normalize:__init()
   parent.__init(self)
end
function L2Normalize:updateOutput(input)
   assert(input:dim() == 2, 'only mini-batch supported (2D tensor), got '
             .. input:dim() .. 'D tensor instead')
   self.output:resizeAs(input)
   local norms = torch.cmul(input,input):sum(2):sqrt()
   self.output:copy(input):cdiv(norms:expandAs(input))
   return self.output
end

function L2Normalize:updateGradInput(input, gradOutput)
   assert(input:dim() == 2, 'only mini-batch supported')
   assert(gradOutput:dim() == 2, 'only mini-batch supported')
   local n = input:size(1) -- batch size
   local d = input:size(2) -- dimensionality of vectors
   
   local sums = torch.sum(torch.cmul(input,input), 2):view(-1)
   local divterms = torch.pow(sums,3/2)
   -- compute diagonal term
   local diag = torch.eye(d):typeAs(input):repeatTensor(n,1):view(n,d,d)
                   :cmul(sums:view(n,1,1):expand(n,d,d))
   -- compute cross term
   local b1 = input:reshape(n,d,1)
   local b2 = input:reshape(n,1,d)
   local cross = - torch.bmm(b1,b2)
   -- compute the local gradient of the L2 transformation
   local dsum = torch.cdiv(diag + cross, divterms:view(n,1,1):expand(n,d,d))
   -- chain the gradient
   self.gradInput = torch.bmm(dsum, gradOutput:view(n,d,1))
   return self.gradInput
end

