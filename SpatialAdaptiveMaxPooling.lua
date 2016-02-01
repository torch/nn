local SpatialAdaptiveMaxPooling, parent = torch.class('nn.SpatialAdaptiveMaxPooling', 'nn.Module')

function SpatialAdaptiveMaxPooling:__init(W, H)
   parent.__init(self)
   
   self.W = W
   self.H = H

   self.indices = torch.Tensor()
end

function SpatialAdaptiveMaxPooling:updateOutput(input)
   input.THNN.SpatialAdaptiveMaxPooling_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.indices:cdata(),
      self.W, self.H
   )
   return self.output
end

function SpatialAdaptiveMaxPooling:updateGradInput(input, gradOutput)
   input.THNN.SpatialAdaptiveMaxPooling_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.indices:cdata()
   )
   return self.gradInput
end

function SpatialAdaptiveMaxPooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.indices:resize()
   self.indices:storage():resize(0)
end
