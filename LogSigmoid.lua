local LogSigmoid, parent = torch.class('nn.LogSigmoid', 'nn.Module')

function LogSigmoid:__init()
   parent.__init(self)
   self.buffer = torch.Tensor()
end

function LogSigmoid:updateOutput(input)
   return input.nn.LogSigmoid_updateOutput(self, input)
end

function LogSigmoid:updateGradInput(input, gradOutput)
   return input.nn.LogSigmoid_updateGradInput(self, input, gradOutput)
end
