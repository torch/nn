local LenSoftMax, _ = torch.class('nn.LenSoftMax', 'nn.Module')

function LenSoftMax:updateOutput(input)
   local _input, _len = unpack(input)
   _input.THNN.LenSoftMax_updateOutput(
      _input:cdata(),
      self.output:cdata(),
      _len:cdata()
   )
   return self.output
end

function LenSoftMax:updateGradInput(input, gradOutput)
   local _input, _len = unpack(input)
   _input.THNN.LenSoftMax_updateGradInput(
      _input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.output:cdata(),
      _len:cdata()
   )
   return self.gradInput
end
