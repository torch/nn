local CMul, parent = torch.class('nn.CMul', 'nn.Module')

function CMul:__init(inputSize)
   parent.__init(self)
  
   self.weight = torch.Tensor(inputSize)
   self.gradWeight = torch.Tensor(inputSize)
   
   -- state
   -- self.gradInput:resize(inputSize)
   self.output:resize(inputSize) 

   self:reset()
end
 
function CMul:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(1))
   end
   self.weight:uniform(-stdv,stdv)
end

function CMul:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   if input:nElement() == self.weight:nElement() then
      self.output:view(-1):cmul(self.weight:view(-1));
   else
      if input:isSameSizeAs(self.weight) then
         self.output:cmul(self.weight)
      else
         local batchSize = input:size(1)
         self.output:view(batchSize, -1):cmul(self.weight:view(1,-1):expandAs(input:view(batchSize, -1)))
      end
   end
   return self.output
end

function CMul:updateGradInput(input, gradOutput)
   if self.gradInput then
      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      self.gradInput:zero()
      if self.weight:nElement() == gradOutput:nElement() then
         self.gradInput:addcmul(1, self.weight, gradOutput)
      else
         local gradOutput = gradOutput:view(input:size(1), -1)
         local gradInput = self.gradInput:view(input:size(1), -1)
         gradInput:addcmul(1, self.weight:view(1,-1):expandAs(gradOutput), gradOutput)
      end
      return self.gradInput
   end
end

function CMul:accGradParameters(input, gradOutput, scale)
   if self.weight:nElement() == gradOutput:nElement() then
      self.gradWeight:addcmul(scale or 1, input, gradOutput)
   else
      local batchSize = input:size(1)
      local input = input:view(batchSize, -1)
      local gradOutput = gradOutput:view(batchSize, -1)
      local gradWeight = self.gradWeight:view(1, -1)
      for i=1,batchSize do
         gradWeight:addcmul(scale or 1, input[i], gradOutput[i])
      end
   end
end
