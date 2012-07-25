local TemporalMaxPooling, parent = torch.class('nn.TemporalMaxPooling', 'nn.Module')

function TemporalMaxPooling:__init(kW, dW)
   parent.__init(self)

   dW = dW or kW
   
   self.kW = kW
   self.dW = dW

   self.indices = torch.Tensor()
end

function TemporalMaxPooling:updateOutput(input)
   input.nn.TemporalMaxPooling_updateOutput(self, input)
   return self.output
end

function TemporalMaxPooling:updateGradInput(input, gradOutput)
   input.nn.TemporalMaxPooling_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

function TemporalMaxPooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.indices:resize()
   self.indices:storage():resize(0)
end
