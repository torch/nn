local MixtureTable, parent = torch.class('nn.MixtureTable', 'nn.Module')

function MixtureTable:__init()
   parent.__init(self)
   self._gate = torch.Tensor()
   self.size = torch.LongTensor()
   self.batchSize = 0
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
      end
      self.output:zero()
      for i,expertInput in ipairs(expertInputs) do
         -- multiply each gater output (a gate) by its 
         -- commensurate expert
         self._gate:resize(self.size:storage())
         self._gate:copy(gaterInput:select(2,i))
         self.output:addcmul(expertInput,gate:expandAs(expertInput))
      end
   end
   
   return self.output
end

function MixtureTable:updateGradInput(input, gradOutput)
   
   return self.gradInput
end
