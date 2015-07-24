local ClassNLLCriterion, parent = torch.class('nn.ClassNLLCriterion', 'nn.Criterion')

function ClassNLLCriterion:__init(weights)
   parent.__init(self)
   self.sizeAverage = true
   self.outputTensor = torch.Tensor(1)
   if weights then
       assert(weights:dim() == 1, "weights input should be 1-D Tensor")
       self.weights = weights
   end
end


function ClassNLLCriterion:__len()
   if (self.weights) then
      return #self.weights
   else
      return 0
   end
end


function ClassNLLCriterion:updateOutput(input, target)
   if input:type() == 'torch.CudaTensor' then
      if self.weights == nil then
         -- The CUDA implementation requires self.weights be non-nil
         self.weights = torch.CudaTensor()
      end
      assert(self.weights:dim() == 0 or self.weights:dim() == 1,
         'weights must be 1D or empty')
      -- The cuda code wont check weight size, so we must do it here.
      if self.weights:dim() == 1 then
         if input:dim() == 1 then
            assert(self.weights:size(1) == input:size(1), 
               'Wrong number of weights')
         else
            assert(self.weights:size(1) == input:size(2),
               'Wrong number of weights')
         end
      end
      if input:dim() == 1 then
         self._target = self._target or input.new(1)
         if type(target) == 'number' then
            self._target[1] = target
         else
            self._target:copy(target)
         end
         input.nn.ClassNLLCriterion_updateOutput(self, input, self._target)
      else
         input.nn.ClassNLLCriterion_updateOutput(self, input, target)
      end
      self.output = self.outputTensor[1]
      return self.output
   end

   if input:dim() == 1 then
      if torch.isTensor(target) then target = target[1] end
      self.output = -input[target]
      if self.weights then
          self.output = self.output*self.weights[target]
      end
   elseif input:dim() == 2 then
      local output = 0
      for i=1,target:size(1) do
         if self.weights then
            output = output - input[i][target[i]]*self.weights[target[i]]
         else
            output = output - input[i][target[i]]
         end
      end
      if self.sizeAverage then
         output = output / target:size(1)
      end
      self.output = output
   else
      error('matrix or vector expected')
   end
   return self.output
end

function ClassNLLCriterion:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()

  if input:type() == 'torch.CudaTensor' then
     -- Note: we'll assume that updateOutput() has been called and self.weights
     -- is non-nil.
     if input:dim() == 1 then
        self._target = self._target or input.new(1)
         if type(target) == 'number' then
            self._target[1] = target
         else
            self._target:copy(target)
         end
        input.nn.ClassNLLCriterion_updateGradInput(self, input, self._target)
     else
        input.nn.ClassNLLCriterion_updateGradInput(self, input, target)
     end
     return self.gradInput
  end

  if input:dim() == 1 then
      if torch.isTensor(target) then target = target[1] end
      self.gradInput[target] = -1
      if self.weights then
          self.gradInput[target] = self.gradInput[target]*self.weights[target]
      end
  else
      local z = -1
      if self.sizeAverage then
         z = z / target:size(1)
      end
      for i=1,target:size(1) do
          self.gradInput[i][target[i]] = z
         if self.weights then
             self.gradInput[i][target[i]] = self.gradInput[i][target[i]]*self.weights[target[i]]
         end
      end
   end
   return self.gradInput
end
