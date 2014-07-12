local MixtureTable, parent = torch.class('nn.MixtureTable', 'nn.Module')

function MixtureTable:__init(dim)
   parent.__init(self)
   self.dim = dim
   self._gaterView = torch.Tensor()
   self._expert = torch.Tensor()
   self._expertView = torch.Tensor()
   self._sum = torch.Tensor()
   self.size = torch.LongStorage()
   self.batchSize = 0
   self.gradInput = {torch.Tensor(), {}}
   if self.dim then
      self.gradInput[2] = torch.Tensor()
      self.size2 = torch.LongStorage()
      self._expertView2 = torch.Tensor()
   end
   self.backwardSetup = false
end

function MixtureTable:updateOutput(input) 
   local gaterInput, expertInputs = unpack(input)
   if gaterInput:dim() == 2 then
      if self.dim then -- expertInputs is a Tensor :
         if self.batchSize ~= expertInputs:size(1) then
            self.size:resize(expertInputs:dim()):fill(1)
            self.size[1] = gaterInput:size(1)
            self.size[self.dim] = gaterInput:size(2)
            self.output:resizeAs(expertInputs:select(self.dim, 1))
            self.batchSize = expertInputs:size(1)
            self.backwardSetup = false
         end
         self._gaterView:view(gaterInput, self.size)
         self._expert:cmul(self._gaterView:expandAs(expertInputs), expertInputs)
         self.output:sum(self._expert, self.dim)
         self.output:resizeAs(expertInputs:select(self.dim, 1))
      else             -- expertInputs is a Table :
         if gaterInput:size(2) ~= #expertInputs then
            error"Should be one gater output per expert"
         end
         local expertInput = expertInputs[1]
         if self.batchSize ~= expertInput:size(1) then
            self.size:resize(expertInput:dim()+1):fill(1)
            self.size[1] = gaterInput:size(1)
            self.size[2] = gaterInput:size(2)
            self.output:resizeAs(expertInput)
            self.batchSize = expertInput:size(1)
            self.backwardSetup = false
         end
         self._gaterView:view(gaterInput, self.size)
         self.output:zero()
         -- multiply accumulate gater outputs by their commensurate expert
         for i,expertInput in ipairs(expertInputs) do
            local gate = self._gaterView:select(2,i):expandAs(expertInput)
            self.output:addcmul(expertInput, gate)
         end
      end
   else
      error"Only works with mini-batches"
   end
   return self.output
end

function MixtureTable:updateGradInput(input, gradOutput)
   local gaterInput, expertInputs = unpack(input)
   local gaterGradInput, expertGradInputs = unpack(self.gradInput)
   if gradOutput:dim() == 2 then
      if self.dim then
         if not self.backwardSetup then
            self.size2:resize(expertInputs:dim())
            self.size2:copy(expertInputs:size())
            self.size2[self.dim] = 1
            gaterGradInput:resizeAs(gaterInput)
            self.backwardSetup = true
         end
         
         -- gater updateGradInput
         self._expertView:view(gradOutput, self.size2)
         local gradOutput = self._expertView:expandAs(expertInputs)
         self._expert:cmul(gradOutput, expertInputs)
         self._expertView2:view(self._expert, gaterInput:size(1), gaterInput:size(2), -1)
         gaterGradInput:sum(self._expertView2, 3)
         gaterGradInput:resizeAs(gaterInput)
         
         -- expert updateGradInput
         expertGradInputs:cmul(self._gaterView:expandAs(expertInputs), gradOutput)
      else
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
            --self._expert:resizeAs(expertGradInput)
            self._expert:cmul(gradOutput, expertInputs[i])
            self._expertView:view(self._expert, gradOutput:size(1), -1)
            self._sum:sum(self._expertView, 2)
            gaterGradInput:select(2,i):copy(self._sum:select(2,1))
            
            -- expert updateGradInput
            local gate = self._gaterView:select(2,i):expandAs(expertGradInput)
            expertGradInput:cmul(gate, gradOutput)     
         end
      end
   else
      error"Only works with mini-batches"
   end
   return self.gradInput
end

function MixtureTable:type(type)
   self.output = self.output:type(type)
   self.gradInput[1] = self.gradInput[1]:type(type)
   if self.dim then
      self.gradInput[2] = self.gradInput[2]:type(type)
      self._expertView2 = self._expertView2:type(type)
   else
      for i,expertGradInput in ipairs(self.gradInput[2]) do
         self.gradInput[2][i] = expertGradInput:type(type)
      end
   end
   self._gaterView = self._gaterView:type(type)
   self._expert = self._expert:type(type)
   self._expertView = self._expertView:type(type)
   self._sum = self._sum:type(type)
end
