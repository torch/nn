local Bilinear, parent = torch.class('nn.Bilinear', 'nn.Module')

local function isint(x) return type(x) == 'number' and x == math.floor(x) end
function Bilinear:__assertInput(input)
   assert(input and type(input) == 'table' and #input == 2,
      'input should be a table containing two data Tensors')
   assert(input[1]:nDimension() == 2 and input[2]:nDimension() == 2,
      'input Tensors should be two-dimensional')
   assert(input[1]:size(1) == input[2]:size(1),
      'input Tensors should have the same number of rows (instances)')
   assert(input[1]:size(2) == self.weight:size(2),
      'dimensionality of first input is erroneous')
   assert(input[2]:size(2) == self.weight:size(3),
      'dimensionality of second input is erroneous')
end
function Bilinear:__assertInputGradOutput(input, gradOutput)
   assert(input[1]:size(1) == gradOutput:size(1),
      'number of rows in gradOutput does not match input')
   assert(gradOutput:size(2) == self.weight:size(1),
      'number of columns in gradOutput does not output size of layer')
end

function Bilinear:__init(inputSize1, inputSize2, outputSize, bias)

   -- assertions:
   assert(self and inputSize1 and inputSize2 and outputSize,
      'should specify inputSize1 and inputSize2 and outputSize')
   assert(isint(inputSize1) and isint(inputSize2) and isint(outputSize),
      'inputSize1 and inputSize2 and outputSize should be integer numbers')
   assert(inputSize1 > 0 and inputSize2 > 0 and outputSize > 0,
      'inputSize1 and inputSize2 and outputSize should be positive numbers')

   -- set up model:
   parent.__init(self)
   local bias = ((bias == nil) and true) or bias
   self.weight     = torch.Tensor(outputSize, inputSize1, inputSize2)
   self.gradWeight = torch.Tensor(outputSize, inputSize1, inputSize2)
   if bias then
      self.bias     = torch.Tensor(outputSize)
      self.gradBias = torch.Tensor(outputSize)
   end
   self.gradInput = {torch.Tensor(), torch.Tensor()}
   self:reset()
end

function Bilinear:reset(stdv)
   assert(self)
   if stdv then
      assert(stdv and type(stdv) == 'number' and stdv > 0,
         'standard deviation should be a positive number')
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1 / math.sqrt(self.weight:size(2))
   end
   self.weight:uniform(-stdv, stdv)
   if self.bias then self.bias:uniform(-stdv, stdv) end
   return self
end

function Bilinear:updateOutput(input)
   assert(self)
   self:__assertInput(input)

   -- set up buffer:
   self.buff = self.buff or input[1].new()
   self.buff:resizeAs(input[2])

   -- compute output scores:
   self.output:resize(input[1]:size(1), self.weight:size(1))
   for k = 1,self.weight:size(1) do
      torch.mm(self.buff, input[1], self.weight[k])
      self.buff:cmul(input[2])
      torch.sum(self.output:narrow(2, k, 1), self.buff, 2)
   end
   if self.bias then
       self.output:add(
           self.bias:reshape(1, self.bias:nElement()):expandAs(self.output)
       )
   end
   return self.output
end

function Bilinear:updateGradInput(input, gradOutput)
   assert(self)
   if self.gradInput then
      self:__assertInputGradOutput(input, gradOutput)

      -- compute d output / d input:
      self.gradInput[1]:resizeAs(input[1]):fill(0)
      self.gradInput[2]:resizeAs(input[2]):fill(0)
      if torch.type(self.weight, 'torch.CudaTensor') then
         for k = 1,self.weight:size(1) do
            torch.addmm(self.gradInput[1], input[2], self.weight[k]:t())
            torch.addmm(self.gradInput[2], input[1], self.weight[k])
         end  -- TODO: Remove this once cutorch gets addbmm() function
      else
         self.gradInput[1]:addbmm(
            input[2]:view(1, input[2]:size(1), input[2]:size(2)):expand(
               self.weight:size(1), input[2]:size(1), input[2]:size(2)
            ),
            self.weight:transpose(2, 3)
         )
         self.gradInput[2]:addbmm(
            input[1]:view(1, input[1]:size(1), input[1]:size(2)):expand(
               self.weight:size(1), input[1]:size(1), input[1]:size(2)
            ), self.weight
         )
      end
      return self.gradInput
   end
end

function Bilinear:accGradParameters(input, gradOutput, scale)
   local scale = scale or 1
   self:__assertInputGradOutput(input, gradOutput)
   assert(scale and type(scale) == 'number' and scale >= 0)

   -- make sure we have buffer:
   self.buff = self.buff or input[1].new()
   self.buff:resizeAs(input[2])

   -- accumulate parameter gradients:
   for k = 1,self.weight:size(1) do
      torch.cmul(
         self.buff, input[1], gradOutput:narrow(2, k, 1):expandAs(input[1])
      )
      self.gradWeight[k]:addmm(self.buff:t(), input[2])
   end
   if self.bias then self.gradBias:add(scale, gradOutput:sum(1)) end
end

-- we do not need to accumulate parameters when sharing:
Bilinear.sharedAccUpdateGradParameters = Bilinear.accUpdateGradParameters

function Bilinear:__tostring__()
  return torch.type(self) ..
      string.format(
         '(%dx%d -> %d) %s',
         self.weight:size(2), self.weight:size(3), self.weight:size(1),
         (self.bias == nil and ' without bias' or '')
      )
end
