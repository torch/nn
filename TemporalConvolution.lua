local TemporalConvolution, parent = torch.class('nn.TemporalConvolution', 'nn.Module')

function TemporalConvolution:__init(inputFrameSize, outputFrameSize, kW, dW)
   parent.__init(self)

   dW = dW or 1

   self.inputFrameSize = inputFrameSize
   self.outputFrameSize = outputFrameSize
   self.kW = kW
   self.dW = dW

   self.weight = torch.Tensor(outputFrameSize, inputFrameSize*kW)
   self.bias = torch.Tensor(outputFrameSize)
   self.gradWeight = torch.Tensor(outputFrameSize, inputFrameSize*kW)
   self.gradBias = torch.Tensor(outputFrameSize)
   
   self:reset()
end

function TemporalConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.inputFrameSize)
   end
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv)
end

function TemporalConvolution:updateOutput(input)
   return input.nn.TemporalConvolution_updateOutput(self, input)
end

function TemporalConvolution:updateGradInput(input, gradOutput)
   if self.gradInput then
      return input.nn.TemporalConvolution_updateGradInput(self, input, gradOutput)
   end
end

function TemporalConvolution:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   input.nn.TemporalConvolution_accGradParameters(self, input, gradOutput, scale)
end

-- we do not need to accumulate parameters when sharing
TemporalConvolution.sharedAccUpdateGradParameters = TemporalConvolution.accUpdateGradParameters
