local WeightedEuclidean, parent = torch.class('nn.WeightedEuclidean', 'nn.Module')

function WeightedEuclidean:__init(inputSize,outputSize)
   parent.__init(self)

   self.templates = torch.Tensor(inputSize,outputSize)
   self.gradTemplates = torch.Tensor(inputSize,outputSize)

   -- each template (output dim) has its own diagonal covariance matrix
   self.diagCov = torch.Tensor(inputSize,outputSize)
   self.gradDiagCov = torch.Tensor(inputSize,outputSize)
   
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
   -- lazy-initialize 
   self._temp = self._temp or self.output.new()
   self._ones = self._ones or self.output.new{1}
   self._diagCov = self._diagCov or self.output.new()
   self._repeat = self._repeat or self.output.new()
   self._sum = self._sum or self.output.new()
   self._temp:resizeAs(input)
   if input:dim() == 1 then
      self.output:resize(self.templates:size(2))
      for outIdx = 1,self.templates:size(2) do
         self._temp:copy(input):add(-1,self.templates:select(2,outIdx))
         self._temp:cmul(self._temp)
         local diagCov = self.diagCov:select(2,outIdx)
         self._temp:cmul(diagCov):cmul(diagCov)
         self.output[outIdx] = math.sqrt(self._temp:sum())
      end
   elseif input:dim() == 2 then
      self.output:resize(input:size(1), self.templates:size(2))
      if self._ones:size(1) ~= input:size(1) then
         self._ones:resize(input:size(1)):fill(1)
      end
      for outIdx = 1,self.templates:size(2) do
         self._temp:copy(input)
         self._temp:addr(-1, self._ones, self.templates:select(2,outIdx))
         self._temp:cmul(self._temp)
         local diagCov = self.diagCov:select(2,outIdx)
         self._diagCov:resizeAs(diagCov):copy(diagCov)
         self._diagCov:pow(2)
         self._diagCov:resize(1,self._diagCov:size(1))
         self._repeat:repeatTensor(self._diagCov, input:size(1), 1)
         self._temp:cmul(self._temp, self._repeat)
         self._sum:sum(self._temp, 2):sqrt()
         self.output:select(2,outIdx):copy(self._sum)
      end
   else
      error"1D or 2D input expected"
   end
   return self.output
end

function WeightedEuclidean:updateGradInput(input, gradOutput)
   self._gradTemp = self._gradTemp or self.output.new()
   self.gradInput:resizeAs(input):zero()
   self._temp:resizeAs(input)
   self._gradTemp:cdiv(gradOutput, self.output)
   if input:dim() == 1 then
      for outIdx = 1,self.templates:size(2) do
         self._temp:copy(input):add(-1,self.templates:select(2,outIdx))
         local diagCov = self.diagCov:select(2,outIdx)
         self._temp:cmul(diagCov):cmul(diagCov)
         
         self._temp:mul(self._gradTemp[outIdx])
         self.gradInput:add(self._temp)
      end
   elseif input:dim() == 2 then
      if self._ones:size(1) ~= input:size(1) then
         self._ones:resize(input:size(1)):fill(1)
      end
      for outIdx = 1,self.templates:size(2) do
         self._temp:copy(input)
         self._temp:addr(-1, self._ones, self.templates:select(2,outIdx))
         local diagCov = self.diagCov:select(2,outIdx)
         self._diagCov:resizeAs(diagCov):copy(diagCov)
         self._diagCov:pow(2)
         self._diagCov:resize(1,self._diagCov:size(1))
         self._repeat:repeatTensor(self._diagCov, input:size(1), 1)
         self._temp:cmul(self._temp, self._repeat)

         local gradTemp = self._gradTemp:select(2, outIdx)
         gradTemp = gradTemp:reshape(1,gradTemp:size(1))
         self._repeat:repeatTensor(gradTemp, input:size(2), 1)
         self.gradInput:addcmul(1, self._temp, self._repeat)
      end
   else
      error"1D or 2D input expected"
   end
   return self.gradInput
end

function WeightedEuclidean:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   self._temp:resizeAs(input)
   self._gradTemp:cdiv(gradOutput, self.output)
   if input:dim() == 1 then
      for outIdx = 1,self.templates:size(2) do
         self._temp:copy(self.templates:select(2,outIdx)):add(-1,input)
         local diagCov = self.diagCov:select(2,outIdx)
         self._temp:cmul(diagCov):cmul(diagCov)
         
         self._temp:mul(self._gradTemp[outIdx])
         self.gradTemplates:select(2,outIdx):add(scale, self._temp)

         self._temp:copy(self.templates:select(2,outIdx)):add(-1,input)
         self._temp:pow(2)
         self._temp:cmul(self.diagCov:select(2,outIdx))
         self._temp:mul(self._gradTemp[outIdx])
         self.gradDiagCov:select(2,outIdx):add(scale, self._temp)
      end
   elseif input:dim() == 2 then
      for outIdx = 1,self.templates:size(2) do
         -- gradTemplates
         self._temp:copy(input)
         self._temp:addr(-1, self._ones, self.templates:select(2,outIdx))
         local diagCov = self.diagCov:select(2,outIdx)
         self._diagCov:resizeAs(diagCov):copy(diagCov)
         self._diagCov:pow(2)
         self._diagCov:resize(1,self._diagCov:size(1))
         self._repeat:repeatTensor(self._diagCov, input:size(1), 1)
         self._temp:cmul(self._temp, self._repeat)
         
         local gradTemp = self._gradTemp:select(2, outIdx)
         gradTemp = gradTemp:reshape(1,gradTemp:size(1))
         self._repeat:repeatTensor(gradTemp, input:size(2), 1)
         self._temp:cmul(self._repeat)
         self._sum:sum(self._temp, 1)
         self.gradTemplates:select(2,outIdx):add(scale, self._sum)

         -- gradDiagCov
         local template = self.templates:select(2,outIdx)
         template = template:reshape(1, template:size(1))
         self._temp:repeatTensor(template, input:size(1), 1)
         self._temp:add(-1,input)
         self._temp:pow(2)
         self._temp:cmul(self._repeat)
         diagCov = diagCov:reshape(1, diagCov:size(1))
         self._repeat:repeatTensor(self._diagCov, input:size(1), 1)
         self._temp:cmul(self._repeat)
         self._sum:sum(self._temp, 1)
         self.gradDiagCov:select(2,outIdx):add(scale, self._sum)
      end
   else
      error"1D or 2D input expected"
   end
end
