local LogSoftMax = torch.class('nn.LogSoftMax', 'nn.Module')

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput then
      if not gradOutput:isContiguous() then
         self._gradOutput = self._gradOutput or gradOutput.new()
         self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
         gradOutput = self._gradOutput
      end
   end
   return input, gradOutput
end

function LogSoftMax:updateOutput(input)
   input = makeContiguous(self, input)

   input.THNN.LogSoftMax_updateOutput(
      input:cdata(),
      self.output:cdata()
   )
   return self.output
end

function LogSoftMax:updateGradInput(input, gradOutput)
   input, gradOutput = makeContiguous(self, input, gradOutput)

   input.THNN.LogSoftMax_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.output:cdata()
   )
   return self.gradInput
end
