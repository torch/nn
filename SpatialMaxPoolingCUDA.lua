local SpatialMaxPoolingCUDA, parent = torch.class('nn.SpatialMaxPoolingCUDA', 'nn.Module')

function SpatialMaxPoolingCUDA:__init(kW, kH, dW, dH)
   parent.__init(self)

   dW = dW or kW
   dH = dH or kH
   
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH
end

function SpatialMaxPoolingCUDA:updateOutput(input)
   input.nn.SpatialMaxPoolingCUDA_updateOutput(self, input)
   return self.output
end

function SpatialMaxPoolingCUDA:updateGradInput(input, gradOutput)
   input.nn.SpatialMaxPoolingCUDA_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

