local Reshape, parent = torch.class('nn.Reshape', 'nn.Module')

function Reshape:__init(...)
   parent.__init(self)
   local arg = {...}

   self.size = torch.LongStorage()
   self.batchsize = torch.LongStorage()
   local n = #arg
   if n == 1 and torch.typename(arg[1]) == 'torch.LongStorage' then
      self.size:resize(#arg[1]):copy(arg[1])
   else
      self.size:resize(n)
      for i=1,n do
         self.size[i] = arg[i]
      end
   end
   self.nelement = 1
   self.batchsize:resize(#self.size+1)
   for i=1,#self.size do
      self.nelement = self.nelement * self.size[i]
      self.batchsize[i+1] = self.size[i]
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
