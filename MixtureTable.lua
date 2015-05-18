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
   self._gradInput = torch.Tensor()
   self.size2 = torch.LongStorage()
   self._expertView2 = torch.Tensor()
   self._expert2 = torch.Tensor()
   self.backwardSetup = false
end

function MixtureTable:updateOutput(input) 
   local gaterInput, expertInputs = table.unpack(input)
   
   self.dimG = 2
   local batchSize = gaterInput:size(1)
   if gaterInput:dim() < 2 then
      self.dimG = 1
      self.dim = self.dim or 1
      batchSize = 1
   end
   self.dim = self.dim or 2
      
   if self.table or torch.type(expertInputs) == 'table' then 
      -- expertInputs is a Table :
      self.table = true
      if gaterInput:size(self.dimG) ~= #expertInputs then
         error"Should be one gater output per expert"
      end
      local expertInput = expertInputs[1]
      if self.batchSize ~= batchSize then
         self.size:resize(expertInput:dim()+1):fill(1)
         if self.dimG > 1 then 
            self.size[1] = gaterInput:size(1)
         end
         self.size[self.dim] = gaterInput:size(self.dimG)
         self.output:resizeAs(expertInput)
         if torch.type(self.gradInput[2]) ~= 'table' then
            self.gradInput[2] = {}
         end
         self.backwardSetup = false
         self.batchSize = batchSize
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
      if self.batchSize ~= batchSize then
         self.size:resize(expertInputs:dim()):fill(1)
         if self.dimG > 1 then
            self.size[1] = gaterInput:size(1)
         end
         self.size[self.dim] = gaterInput:size(self.dimG)
         self.output:resizeAs(expertInputs:select(self.dim, 1))
         self.gradInput[2] = self._gradInput
         self.batchSize = batchSize
         self.backwardSetup = false
      end
      self._gaterView:view(gaterInput, self.size)
      self._expert:cmul(self._gaterView:expandAs(expertInputs), expertInputs)
      self.output:sum(self._expert, self.dim)
      self.output:resizeAs(expertInputs:select(self.dim, 1))
   end

   return self.output
end

function MixtureTable:updateGradInput(input, gradOutput)
   local gaterInput, expertInputs = table.unpack(input)
   local gaterGradInput, expertGradInputs = table.unpack(self.gradInput)
      
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
         self._expert:cmul(gradOutput, expertInputs[i])
         if self.dimG == 1 then
            self._expertView:view(self._expert, -1)
         else
            self._expertView:view(self._expert, gradOutput:size(1), -1)
         end
         self._sum:sum(self._expertView, self.dimG)
         if self.dimG == 1 then
            gaterGradInput[i] = self._sum:select(self.dimG,1)
         else
            gaterGradInput:select(self.dimG,i):copy(self._sum:select(self.dimG,1))
         end
         
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
      local expert = self._expert:transpose(self.dim, self.dimG)
      if not expert:isContiguous() then
         self._expert2:resizeAs(expert)
         self._expert2:copy(expert)
         expert = self._expert2
      end
      if self.dimG == 1 then
         self._expertView2:view(expert, gaterInput:size(1), -1)
      else
         self._expertView2:view(expert, gaterInput:size(1), gaterInput:size(2), -1)
      end
      gaterGradInput:sum(self._expertView2, self.dimG+1)
      gaterGradInput:resizeAs(gaterInput)
      
      -- expert updateGradInput
      expertGradInputs:cmul(self._gaterView:expandAs(expertInputs), gradOutput)
   end

   return self.gradInput
end
