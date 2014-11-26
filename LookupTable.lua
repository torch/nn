local LookupTable, parent = torch.class('nn.LookupTable', 'nn.Module')

LookupTable.__version = 3

function LookupTable:__init(nIndex, ...)
   parent.__init(self)
   local arg = {...}

   if select('#', ...) == 1 and type(arg[1]) ~= "number" then
      local size = arg[1]
      self.size = torch.LongStorage(#size + 1)
      for i=1,#size do
         self.size[i+1] = size[i]
      end
   else
      self.size = torch.LongStorage(select('#', ...)+1)
      for i=1,select('#',...) do
         self.size[i+1] = arg[i]
      end
   end

   self.size[1] = nIndex
   
   local batchSize = torch.LongTensor(#self.size + 1)
   batchSize:narrow(1, 2,#self.size):copy(torch.LongTensor(self.size))
   batchSize[1] = 1
   self.batchSize = batchSize:storage()
   
   self.weight = torch.Tensor(self.size)
   self.gradWeight = torch.Tensor(self.size):zero()
   self.inputs = {}
   
   self.accUpdate = false

   self.nBackward = 0
   self:reset()
end

function LookupTable:accUpdateOnly()
   self.accUpdate = true
   self.gradWeight = nil
end

function LookupTable:reset(stdv)
   stdv = stdv or 1
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.normal(0, stdv)
      end)
   else
      self.weight:normal(0, stdv)
   end
end

function LookupTable:updateOutput(input)
   -- make sure input is a contiguous torch.LongTensor
   if (not input:isContiguous()) or torch.type(input) ~= 'torch.LongTensor' then
      self._indices = self._indices or torch.LongTensor()
      self._indices:resize(input:size()):copy(input)
      input = self._indices
   end
   
   if input:dim() == 1 then
      local nIndex = input:size(1)
      self.size[1] = nIndex
      self.output:index(self.weight, 1, input)
   elseif input:dim() == 2 then
      local nExample = input:size(1)
      local nIndex = input:size(2)
      self.batchSize[1] = nExample
      self.batchSize[2] = nIndex
      
      self._inputView = self._inputView or torch.LongTensor()
      self._inputView:view(input, -1)
      self.output:index(self.weight, 1, self._inputView)
      self.output = self.output:view(nExample, nIndex, self.size[2])
   end

   return self.output
end

function LookupTable:zeroGradParameters()
   if not self.accUpdate then
      for k,_ in pairs(self.inputs) do
         self.gradWeight:select(1, k):zero()
      end
   end
   self.inputs = {}
   self.nBackward = 0
end

function LookupTable:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.nBackward = self.nBackward + 1
      for i=1,input:size(1) do
         local k = input[i]
         self.inputs[k] = (self.inputs[k] or 0) + 1
         self.gradWeight:select(1, k):add(scale, gradOutput:select(1, i))
      end
   elseif input:dim() == 2 then
      self.nBackward = self.nBackward + input:size(1)
      for i=1,input:size(1) do
         local input = input:select(1, i)
         local gradOutput = gradOutput:select(1, i)
         for j=1,input:size(1) do
            local k = input[j]
            self.inputs[k] = (self.inputs[k] or 0) + 1
            self.gradWeight:select(1, k):add(scale, gradOutput:select(1, j))
         end
      end
   end
end

function LookupTable:accUpdateGradParameters(input, gradOutput, lr)
   if input:dim() == 1 then
      for i=1,input:size(1) do
         local k = input[i]
         local kscale = self:scaleUpdateByKey(k)
         self.inputs[k] = (self.inputs[k] or 0) + 1
         self.weight:select(1, input[i]):add(-lr*kscale, gradOutput:select(1, i))
      end
   elseif input:dim() == 2 then 
      for i=1,input:size(1) do
         local input = input:select(1, i)
         local gradOutput = gradOutput:select(1, i)
         for j=1,input:size(1) do
            local k = input[j]
            local kscale = self:scaleUpdateByKey(k)
            self.inputs[k] = (self.inputs[k] or 0) + 1
            self.weight:select(1, k):add(-lr*kscale, gradOutput:select(1, j))
         end
      end
   end
end

function LookupTable:updateParameters(learningRate)
   assert(not self.accUpdate, "use accUpdateGradParameters instead")
   for k,nBackward in pairs(self.inputs) do
      local kscale = self:scaleUpdateByKey(k)
      self.weight:select(1, k):add(-learningRate*kscale, self.gradWeight:select(1, k))
   end
end

function LookupTable:type(type)
   self._indices = nil
   self._inputView = nil
   parent.type(self, type)
end

-- scale the update for each key
function LookupTable:scaleUpdateByKey(inputKey)
   -- default is to perform no key-based scalling
   return 1
end

-- we do not need to accumulate parameters when sharing
LookupTable.sharedAccUpdateGradParameters = LookupTable.accUpdateGradParameters
