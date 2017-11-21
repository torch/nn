local LenSoftMax, parent = torch.class('nn.LenSoftMax', 'nn.Module')

function LenSoftMax:__init()
   parent.__init(self)
   self.gradInput = {torch.Tensor()}
end

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
      self.gradInput[1]:cdata(),
      self.output:cdata(),
      _len:cdata()
   )
   if not self.gradInput[2] then
      self.gradInput[2] = _len.new()
   end
   self.gradInput[2]:resizeAs(_len):zero()
   return self.gradInput
end
