local MixtureTable, parent = torch.class('nn.MixtureTable', 'nn.Module')

function MixtureTable:__init()
   parent.__init(self)
   self._gaterView = torch.Tensor()
   self._expert = torch.Tensor()
   self._expertView = torch.Tensor()
   self.size = torch.LongTensor()
   self.batchSize = 0
   self.gradInput = {torch.Tensor(), {}}
   self.backwardSetup = false
end

function MixtureTable:updateOutput(input) 
   local gaterInput, expertInputs = unpack(input)
   if gaterInput:dim() == 2 then
      if gaterInput:size(2) ~= #expertInputs then
         error"Should be one gater output per expert"
      end
      local expertInput = expertInputs[1]
      if self.batchSize ~= expertInput:size(1) then
         self.size:resize(expertInput:dim()):fill(1)
         self.size[1] = expertInput:size(1)
         self.output:resizeAs(expertInput)
         self.batchSize ~= expertInput:size(1)
         self.backwardSetup = false
      end
      
      self.output:zero()
      -- multiply accumulate gater outputs by their commensurate expert
      for i,expertInput in ipairs(expertInputs) do
         self._expert:view
         self._gaterView:view(gaterInput:select(2,i), self.size:storage())
         self.output:addcmul(expertInput,self._gaterView:expandAs(expertInput))
      end
   end
   return self.output
end

function MixtureTable:updateGradInput(input, gradOutput)
   local gaterInput, expertInputs = unpack(input)
   local gaterGradInput, expertGradInputs = unpack(self.gradInput)
   if gaterInput:dim() == 2 then
      if not self.backwardSetup then
         for i,expertInput in ipairs(expertInputs) do
            local expertGradInput = expertGradInputs[i] or expertInput:clone()
            expertGradInput:resizeAs(expertInput)
            expertGradInputs[i] = expertGradInput
         end
         gaterGradInput:resizeAs(gaterInput)
         self.backwardSetup = true
      end
      
      -- like CMulTable, but with broadcasting
      for i,expertGradInput in ipairs(expertGradInputs) do
         -- gater updateGradInput
         self._expert:resizeAs(expertGradInput)
         self._expert:cmul(gradOutput, expertInputs[i])
         self._expertView:view(self._expert, gradOutput:size(1), -1)
         gaterGradInput:select(2,i):sum(self._expert, 2)
         
         -- expert updateGradInput
         self._gaterView:view(gaterInput:select(2,i), self.size:storage())
         expertGradInput:cmul(self._gaterView:expandAs(expertGradInput), gradOutput)        
      end
   end
   return self.gradInput
end

function MixtureTable:type(type)
   self.output = self.output:type(type)
   self.gradInput[1] = self.gradInput[1]:type(type)
   for i,expertGradInput in ipairs(self.gradInput[2]) do
      self.gradInput[2][i] = expertGradInput:type(type)
   end
   self._gaterView = self._gaterView:type(type)
   self._expert = self._expert:type(type)
   self._expertView = self._expertView:type(type)
end
