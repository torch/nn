
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
   self.buffer = self.buffer or input.new()
   self.normSquared = self.normSquared or input.new()
   self.normSquared:sum(self.buffer:cmul(input, input), 2)
   self.buffer:sqrt(self.normSquared)
   self.output:copy(input):cdiv(self.buffer:expandAs(input))
   return self.output
end

function L2Normalize:updateGradInput(input, gradOutput)
   assert(input:dim() == 2, 'only mini-batch supported')
   assert(gradOutput:dim() == 2, 'only mini-batch supported')
   local n = input:size(1) -- batch size
   local d = input:size(2) -- dimensionality of vectors
   -- compute diagonal term
   self.eye = self.eye or torch.eye(d):typeAs(input):repeatTensor(n,1):view(n,d,d)
   self.diag = self.diag or self.eye.new()
   self.diag:cmul(self.eye, self.normSquared:view(n,1,1):expand(n,d,d))
   -- compute cross term
   local b1 = input:view(n,d,1)
   local b2 = input:view(n,1,d)
   self.diag:add(-torch.bmm(b1,b2))
   -- compute the local gradient of the L2 transformation
   self.diag:cdiv(torch.pow(self.buffer,3):view(n,1,1):expand(n,d,d))
   -- chain the gradient
   self.gradInput:resize(n,d,1):bmm(self.diag, gradOutput:view(n,d,1)):resize(n,d)
   return self.gradInput
end
