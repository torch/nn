local SpatialAdaptiveMaxPooling, parent = torch.class('nn.SpatialAdaptiveMaxPooling', 'nn.Module')

function SpatialAdaptiveMaxPooling:__init(W, H)
   parent.__init(self)
   
   self.W = W
   self.H = H

   self.indices = torch.Tensor()
end

function SpatialAdaptiveMaxPooling:updateOutput(input)
   input.nn.SpatialAdaptiveMaxPooling_updateOutput(self, input)
   return self.output
end

function SpatialAdaptiveMaxPooling:updateGradInput(input, gradOutput)
   input.nn.SpatialAdaptiveMaxPooling_updateGradInput(self, input, gradOutput)
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
