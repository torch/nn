local TemporalSubSampling, parent = torch.class('nn.TemporalSubSampling', 'nn.Module')

function TemporalSubSampling:__init(inputFrameSize, kW, dW)
   parent.__init(self)

   dW = dW or 1

   self.inputFrameSize = inputFrameSize
   self.kW = kW
   self.dW = dW

   self.weight = torch.Tensor(inputFrameSize)
   self.bias = torch.Tensor(inputFrameSize)
   self.gradWeight = torch.Tensor(inputFrameSize)
   self.gradBias = torch.Tensor(inputFrameSize)
   
   self:reset()
end

function TemporalSubSampling:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW)
   end
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv)
end

function TemporalSubSampling:updateOutput(input)
   return input.nn.TemporalSubSampling_updateOutput(self, input)
end

function TemporalSubSampling:updateGradInput(input, gradOutput)
   if self.gradInput then
      return input.nn.TemporalSubSampling_updateGradInput(self, input, gradOutput)
   end
end

function TemporalSubSampling:accGradParameters(input, gradOutput, scale)
   return input.nn.TemporalSubSampling_accGradParameters(self, input, gradOutput, scale)
end
