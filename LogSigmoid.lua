local LogSigmoid, parent = torch.class('nn.LogSigmoid', 'nn.Module')

function LogSigmoid:__init()
   parent.__init(self)
   self.buffer = torch.Tensor()
end

function LogSigmoid:updateOutput(input)
   input.THNN.LogSigmoid_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.buffer:cdata()
   )
   return self.output
end

function LogSigmoid:updateGradInput(input, gradOutput)
   input.THNN.LogSigmoid_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.buffer:cdata()
   )
   return self.gradInput
end
