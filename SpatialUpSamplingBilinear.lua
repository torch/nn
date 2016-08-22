require 'nn.THNN'
local SpatialUpSamplingBilinear, parent =
   torch.class('nn.SpatialUpSamplingBilinear', 'nn.Module')

--[[
Applies a 2D bilinear up-sampling over an input image composed of several
input planes.

The Y and X dimensions are assumed to be the last 2 tensor dimensions.  For
instance, if the tensor is 4D, then dim 3 is the y dimension and dim 4 is the x.
scale_factor is assumed to be a positive integer.

owidth  = (width-1)*(scale_factor-1) + width
oheight  = (height-1)*(scale_factor-1) + height
--]]

function SpatialUpSamplingBilinear:__init(scale_factor)
   parent.__init(self)

   self.scale_factor = scale_factor
   if self.scale_factor < 1 then
     error('scale_factor must be greater than 1')
   end
   if math.floor(self.scale_factor) ~= self.scale_factor then
     error('scale_factor must be integer')
   end
   self.inputSize = torch.LongStorage(4)
   self.outputSize = torch.LongStorage(4)
end

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

function SpatialUpSamplingBilinear:updateOutput(input)
   assert(input:dim() == 4 or input:dim()==3,
            'SpatialUpSamplingBilinear only support 3D or 4D tensors' )
   local inputwas3D = false
   if input:dim() == 3 then
      input=input:view(-1, input:size(1), input:size(2), input:size(3))
      inputwas3D = true
   end
   input = makeContiguous(self, input)
   assert(input:dim() == 4)
   -- Copy the input size
   local xdim = input:dim()
   local ydim = input:dim() - 1
   for i = 1, input:dim() do
     self.inputSize[i] = input:size(i)
     self.outputSize[i] = input:size(i)
   end
   self.outputSize[ydim] = (self.outputSize[ydim]-1) * (self.scale_factor-1)
                           + self.outputSize[ydim]
   self.outputSize[xdim] = (self.outputSize[xdim]-1) * (self.scale_factor -1)
                           + self.outputSize[xdim]
   -- Resize the output if needed
   self.output:resize(self.outputSize)
   input.THNN.SpatialUpSamplingBilinear_updateOutput(
      input:cdata(),
      self.output:cdata()
   )
   if inputwas3D then
      input = input:squeeze(1)
      self.output = self.output:squeeze(1)
   end
   return self.output
end

function SpatialUpSamplingBilinear:updateGradInput(input, gradOutput)
   assert(input:dim() == 4 or input:dim()==3,
            'SpatialUpSamplingBilinear only support 3D or 4D tensors' )
   assert(input:dim() == gradOutput:dim(),
            'Input and gradOutput should be of same dimension' )
   local inputwas3D = false
   if input:dim() == 3 then
      input=input:view(-1, input:size(1), input:size(2), input:size(3))
      gradOutput=gradOutput:view(-1, gradOutput:size(1), gradOutput:size(2),
                                 gradOutput:size(3))
      inputwas3D = true
   end
   assert(input:dim() == 4 and gradOutput:dim() == 4)
   self.gradInput:resizeAs(input)
   input.THNN.SpatialUpSamplingBilinear_updateGradInput(
      gradOutput:cdata(),
      self.gradInput:cdata()
   )
   if inputwas3D then
      input = input:squeeze(1)
      gradOutput = gradOutput:squeeze(1)
      self.gradInput = self.gradInput:squeeze(1)
   end
   return self.gradInput
end


function SpatialUpSamplingBilinear:__tostring__()
   local s = string.format('%s(%d)', torch.type(self), self.scale_factor)
   return s
end
