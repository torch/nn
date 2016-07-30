local Bottle, parent = torch.class("nn.Bottle", "nn.Container")

function Bottle:__init(module, nInputDim, nOutputDim)
   parent.__init(self)
   self.nInputDim = nInputDim or 2
   self.nOutputDim = nOutputDim or self.nInputDim
   self.dimDelta = self.nInputDim-self.nOutputDim
   -- Used to reshape the gradients
   self.shape = torch.Tensor(self.nInputDim)
   self.outShape = torch.Tensor(self.nOutputDim)
   self.size = nil
   -- add module to modules
   self.modules[1] = module
end

local function inShape(input)
   local size = input:size()
   local output = torch.LongTensor(#size)
   for i=1,#size do
      output[i] = size[i]
   end
   return output
end

function Bottle:updateOutput(input)
   local idx = input:dim()-self.nInputDim+1
   -- see if bottle is required
   if idx > 1 then
      -- bottle the first dims
      local size = inShape(input)
      self.size = input:size()
      local shape = size[{{idx,size:size(1)}}]
      self.shape:copy(shape)
      local batchSize = size[{{1,idx-1}}]:prod()
      self.shape[{{1}}]:mul(batchSize)
      -- Forward with the module's dimension
      local output = self.modules[1]:updateOutput(input:view(unpack(torch.totable(self.shape))))
      assert(output:dim() == self.nOutputDim, "Wrong nr of output dims on module, nr: "..tostring(not output or output:dim()))
      self.outShape:copy(inShape(output))
      if math.abs(self.dimDelta)>0 then
         size:resize(size:size(1)-self.dimDelta)
      end
      size[{{idx,size:size(1)}}]:copy(self.outShape)
      size[{{idx}}]:div(batchSize)
      -- unbottle
      self.output = output:view(unpack(torch.totable(size)))
   else
     self.output = self.modules[1]:updateOutput(input)
   end
   return self.output
end

function Bottle:updateGradInput(input, gradOutput)
   if input:dim()>self.nInputDim then
      self.modules[1]:updateGradInput(input:view(unpack(torch.totable(self.shape))), gradOutput:view(unpack(torch.totable(self.outShape))))
      self.gradInput = self.modules[1].gradInput:view(self.size)
   else
      self.gradInput = self.modules[1]:updateGradInput(input)
   end
   return self.gradInput
end

function Bottle:accGradParameters(input, gradOutput, scale)
   if input:dim()>self.nInputDim then
      self.modules[1]:accGradParameters(input:view(unpack(torch.totable(self.shape))), gradOutput:view(unpack(torch.totable(self.outShape))), scale)
   else
      self.modules[1]:accGradParameters(input, gradOutput, scale)
   end
end
