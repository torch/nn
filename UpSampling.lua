require 'nn.THNN'
local UpSampling, parent =
   torch.class('nn.UpSampling', 'nn.Module')

--[[
Upsamples a given 2D (spatial) or 3D (volumetric) input using either nearest neighbor, or linear
interpolation.

The input data is assumed to be of the form `minibatch x channels x [depth] x height x width`.
Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.

The input parameter scale_factor specifies the amount of upsampling, and is assumed to be a positive
integer. An optional mode parameter specifies either 'nearest' (the default) or 'linear'. Linear refers
to either bilinear for spatial (4D) tensors, or trilinear for volumetric (5D) tensors.

For nearest neighbour, output size will be:

odepth  = depth*scale_factor
owidth  = width*scale_factor
oheight  = height*scale_factor

For linear interpolation:

owidth  = (width-1)*(scale_factor-1) + width
owidth  = (width-1)*(scale_factor-1) + width
oheight  = (height-1)*(scale_factor-1) + height

Alternatively for bilinear or trilinear, [odepth], owidth and oheight can be directly provided as input
--]]

function UpSampling:__init(params, mode)
   parent.__init(self)

   -- Any ambigious mode will default to nearest
   if mode ~= nil and (mode == 'linear' or mode == 'bilinear' or mode == 'trilinear') then
      self.mode = 'linear'
   else
      self.mode = 'nearest'
   end

   self.odepth, self.owidth, self.oheight, self.scale_factor = nil, nil, nil, nil
   if torch.type(params) == 'table' then
      if self.mode == 'nearest' then
         error ('Nearest neighbour upsampling requires a scale_factor')
      end
      self.odepth, self.owidth, self.oheight = params.odepth, params.owidth, params.oheight
      if self.owidth == nil or self.oheight == nil then
         error('Output height and width parameters are required')
      end
   else
      self.scale_factor = params   
      if self.scale_factor < 1 then
         error('scale_factor must be greater than 1')
      end
      if math.floor(self.scale_factor) ~= self.scale_factor then
         error('scale_factor must be integer')
      end
   end

   self.inputSize = torch.LongStorage(5):fill(0)
   self.outputSize = torch.LongStorage(5):fill(0)
end

function UpSampling:setSize(input)
   local xdim = input:dim()
   local ydim = xdim - 1

   local zdim = nil
   if xdim > 4 then
      zdim = xdim - 2
   end

   for i = 1, input:dim() do
      self.inputSize[i] = input:size(i)
      self.outputSize[i] = input:size(i)
   end
   if self.scale_factor ~= nil then
      if zdim ~= nil then
         self.outputSize[zdim] = self.outputSize[zdim] * self.scale_factor
      end
      self.outputSize[ydim] = self.outputSize[ydim] * self.scale_factor
      self.outputSize[xdim] = self.outputSize[xdim] * self.scale_factor
   else
      if zdim ~= nil then
         -- Runtime chech that depth was supplied given received 5D input
         if self.odepth == nil then
            error ('No output depth dimension was supplied for volumetric upsampling')
         end
         self.outputSize[zdim] = self.odepth
      end
      self.outputSize[ydim] = self.oheight
      self.outputSize[xdim] = self.owidth
   end
end

function UpSampling:updateOutput(input)
   local nDim = input:dim()
   if nDim < 4 or nDim > 5 then
      error('UpSampling only supports 4D or 5D tensors')
   end
   local xdim = nDim
   local ydim = xdim - 1
   local zdim
   if nDim == 5 then
      zdim = xdim - 2
   end   
   self:setSize(input)
   if nDim == 4 then
      if self.mode == 'nearest' then
         input.THNN.SpatialUpSamplingNearest_updateOutput(
            input:cdata(),
            self.output:cdata(),
            self.scale_factor
         )
      else
         input.THNN.SpatialUpSamplingBilinear_updateOutput(
            input:cdata(),
            self.output:cdata(),
            self.outputSize[ydim],
            self.outputSize[xdim]
         )
      end
   else
      if self.mode == 'nearest' then
         input.THNN.VolumetricUpSamplingNearest_updateOutput(
            input:cdata(),
            self.output:cdata(),
            self.scale_factor
         )
      else
         input.THNN.VolumetricUpSamplingTrilinear_updateOutput(
            input:cdata(),
            self.output:cdata(),
            self.outputSize[zdim],
            self.outputSize[ydim],
            self.outputSize[xdim]
         )
      end
   end
   return self.output
end

function UpSampling:updateGradInput(input, gradOutput)
   local nDim = input:dim()
   if nDim < 4 or nDim > 5 then
      error('UpSampling only supports 4D or 5D tensors')
   end
   if nDim ~= gradOutput:dim() then
      error('Input and gradOutput should be of same dimension')
   end
   local xdim = nDim
   local ydim = xdim - 1
   local zdim
   if nDim == 5 then
      zdim = xdim - 2
   end   
   self.gradInput:resizeAs(input) 
   if nDim == 4 then
      if self.mode == 'nearest' then
         input.THNN.SpatialUpSamplingNearest_updateGradInput(
            input:cdata(),
            gradOutput:cdata(),
            self.gradInput:cdata(),
            self.scale_factor
         )
      else
         input.THNN.SpatialUpSamplingBilinear_updateGradInput(
            gradOutput:cdata(),
            self.gradInput:cdata(),
            input:size(1),
            input:size(2),
            input:size(3),
            input:size(4),
            self.outputSize[ydim],
            self.outputSize[xdim]
         )
      end
   else
      if self.mode == 'nearest' then
         input.THNN.VolumetricUpSamplingNearest_updateGradInput(
            input:cdata(),
            gradOutput:cdata(),
            self.gradInput:cdata(),
            self.scale_factor
         )
      else
         input.THNN.VolumetricUpSamplingTrilinear_updateGradInput(
            gradOutput:cdata(),
            self.gradInput:cdata(),
            input:size(1),
            input:size(2),
            input:size(3),
            input:size(4),
            input:size(5),
            self.outputSize[zdim],
            self.outputSize[ydim],
            self.outputSize[xdim]
         )
      end
   end
   return self.gradInput
end

function UpSampling:__tostring__()
   local s
   if self.scale_factor ~= nil then
      s = string.format('%s(%dx, %s)', torch.type(self), self.scale_factor, self.mode)
   else
      if self.odepth ~= nil then
         s = string.format('%s(%dx%dx%d, %s)', torch.type(self), self.odepth, self.oheight, self.owidth, self.mode)
      else
         s = string.format('%s(%dx%d, %s)', torch.type(self), self.oheight, self.owidth, self.mode)
      end
   end
   return s
end
