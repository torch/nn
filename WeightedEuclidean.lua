local WeightedEuclidean, parent = torch.class('nn.WeightedEuclidean', 'nn.Module')

function WeightedEuclidean:__init(inputSize,outputSize)
   parent.__init(self)

   self.templates = torch.Tensor(inputSize,outputSize)
   self.gradTemplates = torch.Tensor(inputSize,outputSize)

   self.diagCov = torch.Tensor(inputSize,outputSize)
   self.gradDiagCov = torch.Tensor(inputSize,outputSize)

   self.gradInput:resize(inputSize)
   self.output:resize(outputSize)
   self.temp = torch.Tensor(inputSize)

   -- for compat with Torch's modules (it's bad we have to do that)
   do
      self.weight = self.templates
      self.gradWeight = self.gradTemplates
      self.bias = self.diagCov
      self.gradBias = self.gradDiagCov
   end

   self:reset()
end

function WeightedEuclidean:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.templates:size(1))
   end
   if nn.oldSeed then
      for i=1,self.templates:size(2) do
         self.templates:select(2, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
      end
   else
      self.templates:uniform(-stdv, stdv)
   end
   self.diagCov:fill(1)
end

function WeightedEuclidean:updateOutput(input)
   self.output:zero()
   for o = 1,self.templates:size(2) do
      self.temp:copy(input):add(-1,self.templates:select(2,o))
      self.temp:cmul(self.temp)
      self.temp:cmul(self.diagCov:select(2,o)):cmul(self.diagCov:select(2,o))
      self.output[o] = math.sqrt(self.temp:sum())
   end
   return self.output
end

function WeightedEuclidean:updateGradInput(input, gradOutput)
   self:forward(input)
   self.gradInput:zero()
   for o = 1,self.templates:size(2) do
      if self.output[o] ~= 0 then
         self.temp:copy(input):add(-1,self.templates:select(2,o))
         self.temp:cmul(self.diagCov:select(2,o)):cmul(self.diagCov:select(2,o))
         self.temp:mul(gradOutput[o]/self.output[o])
         self.gradInput:add(self.temp)
      end
   end
   return self.gradInput
end

function WeightedEuclidean:accGradParameters(input, gradOutput, scale)
   self:forward(input)
   scale = scale or 1
   for o = 1,self.templates:size(2) do
      if self.output[o] ~= 0 then
         self.temp:copy(self.templates:select(2,o)):add(-1,input)
         self.temp:cmul(self.diagCov:select(2,o)):cmul(self.diagCov:select(2,o))
         self.temp:mul(gradOutput[o]/self.output[o])
         self.gradTemplates:select(2,o):add(self.temp)

         self.temp:copy(self.templates:select(2,o)):add(-1,input)
         self.temp:cmul(self.temp)
         self.temp:cmul(self.diagCov:select(2,o))
         self.temp:mul(gradOutput[o]/self.output[o])
         self.gradDiagCov:select(2,o):add(self.temp)
      end
   end
end
