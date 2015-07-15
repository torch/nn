local Narrow, parent = torch.class('nn.Narrow', 'nn.Module')

function Narrow:__init(dimension,offset,length)
   parent.__init(self)
   self.dimension=dimension
   self.index=offset
   self.length=length or 1
   if not dimension or not offset then
      error('nn.Narrow(dimension, offset, length)')
   end
end

function Narrow:updateOutput(input)
   local output=input:narrow(self.dimension,self.index,self.length)
   self.output = self.output:typeAs(output)
   self.output:resizeAs(output):copy(output)
   return self.output
end

function Narrow:updateGradInput(input, gradOutput)
   self.gradInput = self.gradInput:typeAs(input)
   self.gradInput:resizeAs(input):zero()
   self.gradInput:narrow(self.dimension,self.index,self.length):copy(gradOutput)
   return self.gradInput
end
