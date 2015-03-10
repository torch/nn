local View, parent = torch.class('nn.View', 'nn.Module')

function View:__init(...)
   parent.__init(self)
   self.size = ...
   if select('#', ...) > 1 or type(self.size) == "number" then
      self.size = torch.LongStorage({...})
   end
   assert(torch.typename(self.size)=="torch.LongStorage", "expecting a LongStorage")
   self.numElements = 1
   for i = 1,#self.size do
      self.numElements = self.numElements * self.size[i]
   end

   self.output = nil
   self.gradInput = nil
   self.numInputDims = nil
end

function View:setNumInputDims(numInputDims)
   self.numInputDims = numInputDims
   return self
end

local function batchsize(input, size, numInputDims, numElements)

   -- handle special vector case
   if size:size() == 1 and size[1] == -1 then
      if numInputDims then
         numElements = 1
         local dim = input:nDimension()
         for i=1,numInputDims do
            numElements = numElements * input:size(dim-numElements+1)
         end
      else
         numElements = input:nElement()
      end
      size = torch.LongStorage{numElements}
   end

   -- find if number of elements is divisible with desired number
   local ine = input:nElement()
   local dim = 0
   local bsz = 1
   while ine > numElements do
      dim = dim + 1
      local dimsz = input:size(dim)
      if ine % numElements == 0 then
         dimsz = math.min(ine/numElements, dimsz)
      end
      ine = ine / dimsz
      bsz = bsz * dimsz
   end

   if ine ~= numElements then
      error(string.format(
               'input view (%s) and desired view (%s) do not match',
               table.concat(input:size():totable(), 'x'),
               table.concat(size:totable(), 'x')))
   end

   if bsz == 1 and (not numInputDims or input:nDimension() <= numInputDims) then
      return
   end

   return bsz
end

function View:updateOutput(input)
   local bsz = batchsize(input, self.size, self.numInputDims, self.numElements)
   if bsz then
      self.output = input:view(bsz, unpack(self.size:totable()))
   else
      self.output = input:view(self.size)
   end
   return self.output
end

function View:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput:view(input:size())
   return self.gradInput
end
