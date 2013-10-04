local SpatialMaxPooling, parent = torch.class('nn.SpatialMaxPooling', 'nn.Module')

function SpatialMaxPooling:__init(kW, kH, dW, dH)
   parent.__init(self)

   dW = dW or kW
   dH = dH or kH
   
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH

   self.indices = torch.Tensor()
end

function SpatialMaxPooling:updateOutput(input)
   input.nn.SpatialMaxPooling_updateOutput(self, input)
   return self.output
end

function SpatialMaxPooling:updateGradInput(input, gradOutput)
   input.nn.SpatialMaxPooling_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

function SpatialMaxPooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.indices:resize()
   self.indices:storage():resize(0)
end
