local TemporalMaxPooling, parent = torch.class('nn.TemporalMaxPooling', 'nn.Module')

function TemporalMaxPooling:__init(kW, dW)
   parent.__init(self)

   dW = dW or kW

   self.kW = kW
   self.dW = dW
end

function TemporalMaxPooling:updateOutput(input)
   self.indices = self.indices or input.new()
   input.THNN.TemporalMaxPooling_updateOutput(
       input:cdata(), self.output:cdata(),
       self.indices:cdata(), self.kW, self.dW
   )
   return self.output
end

function TemporalMaxPooling:updateGradInput(input, gradOutput)
    if self.gradInput then
	input.THNN.TemporalMaxPooling_updateGradInput(
	    input:cdata(), gradOutput:cdata(),
	    self.gradInput:cdata(), self.indices:cdata(),
	    self.kW, self.dW
	)
	return self.gradInput
    end
end

function TemporalMaxPooling:empty()
   self:clearState()
end

function TemporalMaxPooling:clearState()
   if self.indices then self.indices:set() end
   return parent.clearState(self)
end
