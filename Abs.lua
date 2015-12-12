local THNN = require('nn.THNN')

local Abs, parent = torch.class('nn.Abs', 'nn.Module')

function Abs:__init()
   parent.__init(self)
end

function Abs:updateOutput(input)
   THNN.runKernel(
     'Abs_updateOutput',
     input:type(),
     input:cdata(),
     self.output:cdata()
   )
   return self.output
end

function Abs:updateGradInput(input, gradOutput)
   THNN.runKernel(
     'Abs_updateGradInput',
     input:type(),
     input:cdata(),
     gradOutput:cdata(),
     self.gradInput:cdata()
   )
   return self.gradInput
end
