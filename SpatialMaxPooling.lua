local SpatialMaxPooling, parent = torch.class('nn.SpatialMaxPooling', 'nn.Module')

function SpatialMaxPooling:__init(kW, kH, dW, dH, padW, padH)
   parent.__init(self)

   dW = dW or kW
   dH = dH or kH
   
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH

   self.padW = padW or 0
   self.padH = padH or 0

   self.ceil_mode = false
   self.indices = torch.Tensor()
end

function SpatialMaxPooling:ceil()
  self.ceil_mode = true
  return self
end

function SpatialMaxPooling:floor()
  self.ceil_mode = false
  return self
end

function SpatialMaxPooling:updateOutput(input)
   -- backward compatibility
   self.ceil_mode = self.ceil_mode or false
   self.padW = self.padW or 0
   self.padH = self.padH or 0
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

function SpatialMaxPooling:__tostring__()
   local s =  string.format('%s(%d,%d,%d,%d', torch.type(self),
                            self.kW, self.kH, self.dW, self.dH)
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
      s = s .. ',' .. self.padW .. ','.. self.padH
   end
   s = s .. ')'

   return s
end
