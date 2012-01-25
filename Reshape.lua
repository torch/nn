local Reshape, parent = torch.class('nn.Reshape', 'nn.Module')

function Reshape:__init(...)
   parent.__init(self)
   self.size = torch.LongStorage()
   self.batchsize = torch.LongStorage()
   local n = select('#', ...)
   if n == 1 and torch.typename(select(1, ...)) == 'torch.LongStorage' then
      self.size:resize(#select(1, ...)):copy(select(1, ...))
   else
      self.size:resize(n)
      self.batchsize:resize(n+1)
      self.nelement = 1
      for i=1,n do
         self.size[i] = select(i, ...)
         self.batchsize[i+1] = select(i, ...)
         self.nelement = self.nelement * self.size[i]
      end
   end
end

function Reshape:updateOutput(input)
   input = input:contiguous()
   local nelement = input:nElement()
   if nelement == self.nelement then
      self.output:set(input):resize(self.size)
   else
      self.batchsize[1] = input:size(1)
      self.output:set(input):resize(self.batchsize)
   end
   return self.output
end

function Reshape:updateGradInput(input, gradOutput)
   gradOutput = gradOutput:contiguous()
   self.gradInput:set(gradOutput):resizeAs(input)
   return self.gradInput
end
