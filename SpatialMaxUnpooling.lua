local SpatialMaxUnpooling, parent = torch.class('nn.SpatialMaxUnpooling', 'nn.Module')

function SpatialMaxUnpooling:__init(poolingModule)
  parent.__init(self)
  assert(torch.type(poolingModule)=='nn.SpatialMaxPooling', 'Argument must be a nn.SPatialMaxPooling module')
  assert(poolingModule.kH==poolingModule.dH and poolingModule.kW==poolingModule.dW, "The size of pooling module's kernel must be equal to its stride")
  self.pooling = poolingModule

  poolingModule.updateOutput = function(pool, input)
    local dims = input:dim()
    pool.iheight = input:size(dims-1)
    pool.iwidth = input:size(dims)
    return nn.SpatialMaxPooling.updateOutput(pool, input)
  end
end

function SpatialMaxUnpooling:setParams()
  self.indices = self.pooling.indices
  self.oheight = self.pooling.iheight
  self.owidth = self.pooling.iwidth
end

function SpatialMaxUnpooling:updateOutput(input)
  self:setParams()
  input.nn.SpatialMaxUnpooling_updateOutput(self, input)
  return self.output
end

function SpatialMaxUnpooling:updateGradInput(input, gradOutput)
  self:setParams()
  input.nn.SpatialMaxUnpooling_updateGradInput(self, input, gradOutput)
  return self.gradInput
end

function SpatialMaxUnpooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
end

function SpatialMaxUnpooling:__tostring__()
   local s =  string.format('%s(%d,%d,%d,%d', torch.type(self), self.kW, self.kH, self.dW, self.dH)
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
      s = s .. ',' .. self.padW .. ','.. self.padH
   end

   return 'nn.SpatialMaxUnpooling associated to '.. s .. ')'
end
