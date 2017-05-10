local SoftenedSoftMax, parent = torch.class('nn.SoftenedSoftMax', 'nn.Module')

function SoftenedSoftMax:__init(temperature)
   parent.__init(self)
   self.temperature = temperature or 1
   assert(self.temperature  > 0, "Tempareture must be greater than 0")
end

function SoftenedSoftMax:updateOutput(input)
   self.softenedinput = self.softenedinput or input.new()
   self.softenedinput:resizeAs(input):copy(input):div(self.temperature)
   
   input.THNN.SoftMax_updateOutput(
      self.softenedinput:cdata(),
      self.output:cdata()
   )
   return self.output
end

function SoftenedSoftMax:updateGradInput(input, gradOutput)
   self.softenedinput = self.softenedinput or input.new()
   self.softenedinput:resizeAs(input):copy(input):div(self.temperature)
   
   input.THNN.SoftMax_updateGradInput(
      self.softenedinput:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.output:cdata()
   )
   self.gradInput:div(self.temperature)
   return self.gradInput
end

function SoftenedSoftMax:__tostring__()
  return string.format('%s(Temp = %f)', torch.type(self), self.temperature)
end

function SoftenedSoftMax:clearState()
   if self.softenedinput then self.softenedinput:set() end
   return parent.clearState(self)
end
