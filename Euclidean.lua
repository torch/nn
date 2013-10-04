local Euclidean, parent = torch.class('nn.Euclidean', 'nn.Module')

function Euclidean:__init(inputSize,outputSize)
   parent.__init(self)

   self.weight = torch.Tensor(inputSize,outputSize)
   self.gradWeight = torch.Tensor(inputSize,outputSize)

   -- state
   self.gradInput:resize(inputSize)
   self.output:resize(outputSize)
   self.temp = torch.Tensor(inputSize)

   self:reset()
end

function Euclidean:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(1))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(2) do
         self.weight:select(2, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
      end
   else
      self.weight:uniform(-stdv, stdv)
   end
end

function Euclidean:updateOutput(input)
   self.output:zero()
   for o = 1,self.weight:size(2) do
      self.output[o] = input:dist(self.weight:select(2,o))
   end
   return self.output
end

function Euclidean:updateGradInput(input, gradOutput)
   self:updateOutput(input)
   if self.gradInput then
      self.gradInput:zero()
      for o = 1,self.weight:size(2) do
         if self.output[o] ~= 0 then
            self.temp:copy(input):add(-1,self.weight:select(2,o))
            self.temp:mul(gradOutput[o]/self.output[o])
            self.gradInput:add(self.temp)
         end
      end
      return self.gradInput
   end
end

function Euclidean:accGradParameters(input, gradOutput, scale)
   self:updateOutput(input)
   scale = scale or 1
   for o = 1,self.weight:size(2) do
      if self.output[o] ~= 0 then
         self.temp:copy(self.weight:select(2,o)):add(-1,input)
         self.temp:mul(gradOutput[o]/self.output[o])
         self.gradWeight:select(2,o):add(self.temp)
      end
   end
end
