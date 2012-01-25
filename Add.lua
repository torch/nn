local Add, parent = torch.class('nn.Add', 'nn.Module')

function Add:__init(inputSize,scalar)
   parent.__init(self)
  
   local size = inputSize
   if scalar then size=1 end
   self.bias = torch.Tensor(size)
   self.gradBias = torch.Tensor(size)
     
   -- state
   self.gradInput:resize(inputSize)
   self.output:resize(inputSize) 

   self:reset()
end

function Add:reset(stdv)
   if stdv then 
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.bias:size(1))
   end

   for i=1,self.bias:size(1) do
      self.bias[i] = torch.uniform(-stdv, stdv)
   end
end

function Add:updateOutput(input)
   self.output:copy(input);
   if self.gradBias:size(1)==1 then
     self.output:add(self.bias[1]);
   else
     self.output:add(self.bias);
   end
   return self.output
end 

function Add:updateGradInput(input, gradOutput)
   if self.gradInput then
      self.gradInput:copy(gradOutput) 
      return self.gradInput
   end
end

function Add:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if self.gradBias:size(1) == 1 then
      self.gradBias[1] = self.gradBias[1] + scale*gradOutput:sumall();
   else
      self.gradBias:add(scale, gradOutput)
   end
end
