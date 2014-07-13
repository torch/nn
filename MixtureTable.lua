local MixtureTable, parent = torch.class('nn.MixtureTable', 'nn.Module')

function MixtureTable:__init(dim)
   parent.__init(self)
   self.dim = dim or 2
   self._gaterView = torch.Tensor()
   self._expert = torch.Tensor()
   self._expertView = torch.Tensor()
   self._sum = torch.Tensor()
   self.size = torch.LongStorage()
   self.batchSize = 0
   self.gradInput = {torch.Tensor(), {}}
   self._gradInput = torch.Tensor()
   self.size2 = torch.LongStorage()
   self._expertView2 = torch.Tensor()
   self._expert2 = torch.Tensor()
   self.backwardSetup = false
end

function MixtureTable:updateOutput(input) 
   local gaterInput, expertInputs = unpack(input)
   if gaterInput:dim() > 1 then
      if self.table or torch.type(expertInputs) == 'table' then 
         -- expertInputs is a Table :
         self.table = true
         if gaterInput:size(2) ~= #expertInputs then
            error"Should be one gater output per expert"
         end
         local expertInput = expertInputs[1]
         if self.batchSize ~= expertInput:size(1) then
            self.size:resize(expertInput:dim()+1):fill(1)
            self.size[1] = gaterInput:size(1)
            self.size[self.dim] = gaterInput:size(2)
            self.output:resizeAs(expertInput)
            self.batchSize = expertInput:size(1)
            if torch.type(self.gradInput[2]) ~= 'table' then
               self.gradInput[2] = {}
            end
            self.backwardSetup = false
         end
         self._gaterView:view(gaterInput, self.size)
         self.output:zero()
         -- multiply accumulate gater outputs by their commensurate expert
         for i,expertInput in ipairs(expertInputs) do
            local gate = self._gaterView:select(self.dim,i):expandAs(expertInput)
            self.output:addcmul(expertInput, gate)
         end
      else
         -- expertInputs is a Tensor :
         if self.batchSize ~= expertInputs:size(1) then
            self.size:resize(expertInputs:dim()):fill(1)
            self.size[1] = gaterInput:size(1)
            self.size[self.dim] = gaterInput:size(2)
            self.output:resizeAs(expertInputs:select(self.dim, 1))
            self.batchSize = expertInputs:size(1)
            self.gradInput[2] = self._gradInput
            self.backwardSetup = false
         end
         self._gaterView:view(gaterInput, self.size)
         self._expert:cmul(self._gaterView:expandAs(expertInputs), expertInputs)
         self.output:sum(self._expert, self.dim)
         self.output:resizeAs(expertInputs:select(self.dim, 1))
      end
   else
      error"Only works with mini-batches"
   end
   return self.output
end

function MixtureTable:updateGradInput(input, gradOutput)
   local gaterInput, expertInputs = unpack(input)
   local gaterGradInput, expertGradInputs = unpack(self.gradInput)
   if gradOutput:dim() > 1 then
      if self.table then
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
            local gate = self._gaterView:select(self.dim,i):expandAs(expertGradInput)
            expertGradInput:cmul(gate, gradOutput)     
         end
      else
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
         local expert = self._expert:transpose(self.dim, 2)
         if not expert:isContiguous() then
            self._expert2:resizeAs(expert)
            self._expert2:copy(expert)
            expert = self._expert2
         end
         self._expertView2:view(expert, gaterInput:size(1), gaterInput:size(2), -1)
         gaterGradInput:sum(self._expertView2, 3)
         gaterGradInput:resizeAs(gaterInput)
         
         -- expert updateGradInput
         expertGradInputs:cmul(self._gaterView:expandAs(expertInputs), gradOutput)
      end
   else
      error"Only works with mini-batches"
   end
   return self.gradInput
end

function MixtureTable:type(type)
   self.output = self.output:type(type)
   self.gradInput[1] = self.gradInput[1]:type(type)
   if torch.type(self.gradInput[2]) == 'table' then
      for i,expertGradInput in ipairs(self.gradInput[2]) do
         self.gradInput[2][i] = expertGradInput:type(type)
      end
   end
   self._gaterView = self._gaterView:type(type)
   self._expert = self._expert:type(type)
   self._expertView = self._expertView:type(type)
   self._sum = self._sum:type(type)
   self._gradInput = self._gradInput:type(type)
   self._expert2 = self._expert2:type(type)
   self._expertView2 = self._expertView2:type(type)
end
