local Replicate, parent = torch.class('nn.Replicate','nn.Module')

function Replicate:__init(nf, dim)
   parent.__init(self)
   self.nfeatures = nf
   self.dim = dim or 1
   assert(dim > 0, "Can only replicate across positive integer dimensions.")
end

function Replicate:updateOutput(input)
   assert(
      self.dim <= input:dim()+1,
      "Not enough input dimensions to replicate along dimension " ..
      tostring(self.dim) .. ".")
   local sz = torch.LongStorage(input:dim()+1)
   sz[self.dim] = self.nfeatures
   for i = 1,input:dim() do
      local offset = 0
      if i >= self.dim then
         offset = 1
      end
      sz[i+offset] = input:size(i)
   end
   local st = torch.LongStorage(input:dim()+1)
   st[self.dim] = 0
   for i = 1,input:dim() do
      local offset = 0
      if i >= self.dim then
         offset = 1
      end
      st[i+offset] = input:stride(i)
   end
   self.output = input.new(input:storage(),input:storageOffset(),sz,st)
   return self.output
end

function Replicate:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   for k = 1,gradOutput:size(self.dim) do
      self.gradInput:add(gradOutput:select(self.dim, k))
   end
   return self.gradInput
end
