--------------------------------------------------------------------------------
-- rop.lua: This file appends extra methods in nn to calculate the product of
-- the Hessian with a vector, using the R-op. This can be used to quickly
-- estimate curvature in some optimization methods, or can be used to calculate
-- the full Hessian (by repeatedly multiplying with a one-hot vectors).
--
-- To activate this model simply require it. Currently a very limited set of
-- modules have the R-op implemented.
--------------------------------------------------------------------------------
nn = require 'nn'

-- The Module class
local Module = nn.Module

-- Allocate tensors for the R-op applied to the input and the gradient
local Module_init = Module.__init
function Module.__init(self)
   Module_init(self)
   self.rOutput= torch.Tensor()
   self.rGradInput = torch.Tensor()
end

-- Create storage to save the R-op applied to the parameters in
function Module:parameters()
   if self.weight then
      self.rWeight = self.weight:clone():zero()
      self.rGradWeight = self.weight:clone():zero()
   end
   if self.bias then
      self.rBias = self.bias:clone():zero()
      self.rGradBias = self.bias:clone():zero()
   end
   if self.weight and self.bias then
      return {self.weight, self.bias}, {self.gradWeight, self.gradBias},
             {self.rWeight, self.rBias}, {self.rGradWeight, self.rGradBias}
   elseif self.weight then
      return {self.weight}, {self.gradWeight},
             {self.rWeight}, {self.rGradWeight}
   elseif self.bias then
      return {self.bias}, {self.gradBias}, {self.rBias}, {self.rGradBias}
   else
      return
   end
end

-- Flatten the R-op parameters as well
function Module:getParameters()
   local parameters, gradParameters, rParameters,
         rGradParameters = self:parameters()
   return Module.flatten(parameters), Module.flatten(gradParameters),
          Module.flatten(rParameters), Module.flatten(rGradParameters)
end

function Module:rForward(input, rInput)
   -- rInput is zero for the input of the model
   rInput = rInput or input:clone():zero()
   return self:updateROutput(input, rInput)
end

function Module:rBackward(input, rInput, gradOutput, rGradOutput)
   self:updateRGradInput(input, rInput, gradOutput, rGradOutput)
   self:accRGradParameters(input, rInput, gradOutput, rGradOutput)
   return self.rGradInput
end

function Module:accRGradParameters(input, rInput, gradOutput, rGradOutput)
end

-- Container
local Container = nn.Container

function Container:parameters()
    local rval = {}
    for i=1,#self.modules do
        local values = {self.modules[i]:parameters()}
        for j = 1, #values do
           if not rval[j] then rval[j] = {} end
           for k = 1, #values do
              rval[j][#rval[j] + 1] = values[j][k]
           end
        end
    end
    return unpack(rval)
end

-- The Sequential module
local Sequential = nn.Sequential

function Sequential:updateROutput(input, rInput)
   local currentOutput = input
   local currentROutput = rInput
   for i = 1, #self.modules do
      currentROutput = self.modules[i]:updateROutput(
         currentOutput, currentROutput
      )
      currentOutput = self.modules[i].output
   end
   self.rOutput = currentROutput
   return currentROutput
end

function Sequential:updateRGradInput(input, rInput, gradOutput, rGradOutput)
   local currentRGradOutput = rGradOutput
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i = #self.modules-1, 1, -1 do
      local previousModule = self.modules[i]
      currentRGradOutput = currentModule:updateRGradInput(
         previousModule.output, previousModule.rOutput,
         currentGradOutput, currentRGradOutput
      )
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
   end
   self.rGradInput = currentModule:updateRGradInput(
      input, rInput, currentGradOutput, currentRGradOutput
   )
   return self.rGradInput
end

function Sequential:accRGradParameters(input, rInput, gradOutput, rGradOutput)
   local currentRGradOutput = rGradOutput
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   local previousModule = self.modules[#self.modules]
   for i = #self.modules-1, 1, -1 do
      previousModule = self.modules[i]
      currentModule:accRGradParameters(
         previousModule.output, previousModule.rOutput,
         currentGradOutput, currentRGradOutput
      )
      currentGradOutput = currentModule.gradInput
      currentRGradOutput = currentModule.rGradInput
      currentModule = previousModule
   end
   currentModule:accRGradParameters(
      input, rInput, currentGradOutput, currentRGradOutput
   )
   return self.rGradInput
end

-- TODO Why is this needed?
Sequential.parameters = Container.parameters

-- Criterion
---- just for test
local SUMCriterion, parent = torch.class('SUMCriterion', 'nn.Criterion')

function SUMCriterion:__init()
  parent.__init(self)
end
function SUMCriterion:updateOutput(input, target)
  return input:sum()
end

function SUMCriterion:updateGradInput(input, target)
  return torch.ones(input:size())
end

function SUMCriterion:rBackward(rInput, target)
  return self:updateRGradInput(rInput, target)
end

function SUMCriterion:updateRGradInput(rInput, target)
  return torch.zeros(rInput:size())
end
  
  
-- MSE criterion
local Criterion = nn.Criterion
local Criterion_init = Criterion.__init
function Criterion:__init()
   Criterion_init(self)
   self.rGradInput = torch.Tensor()
end
--
--function Criterion:rBackward(input, target)
--end

local MSECriterion = nn.MSECriterion
--function MSECriterion:updateOutput(input, target)
--   self.output = torch.add(input, -target):norm()^2/input:numel()
--   return self.output
--end
--
--function MSECriterion:updateGradInput(input, target)
--   self.gradInput:resizeAs(input)
--   self.gradInput:copy(input)
--   self.gradInput:add(-target)
--   self.gradInput:div(input:numel()/2)
--   return self.gradInput
--end

function MSECriterion:rBackward(rInput, target)
   return self:updateRGradInput(rInput, target)
end

function MSECriterion:updateRGradInput(rInput, target) 
    self.rGradInput:resizeAs(rInput)
    self.rGradInput:copy(rInput)
    self.rGradInput:div(rInput:numel()/2)
    return self.rGradInput
end

--
local ClassNLLCriterion = nn.ClassNLLCriterion
function ClassNLLCriterion:rBackward(rInput, target)
   return self:updateRGradInput(rInput, target)
end

function ClassNLLCriterion:updateRGradInput(rInput, target) 
--    b_size = rInput:size(1) 
    self.rGradInput:resizeAs(rInput)
    self.rGradInput:fill(0) ------
    return self.rGradInput
end

-- The Linear module (for now inputDim <=2 is assumed)
local Linear = nn.Linear

function Linear:updateROutput(input, rInput)
  if input:dim() == 1 then
    self.rOutput:resize(self.bias:size(1))
    self.rOutput:mv(self.rWeight, input)
    self.rOutput:addmv(self.weight, rInput)
    self.rOutput:add(self.rBias)
  elseif  input:dim() == 2 then
    self.rOutput:resize(input:size(1), self.bias:size(1))
    self.rOutput:mm(input, self.rWeight:t()) -- a*V'
    self.rOutput:addmm(rInput, self.weight:t()) -- R(a)*W'

    for i = 1, self.rBias:size(1) do -- add it to the ith column
      self.rOutput[{{},{i}}]:add(self.rBias[i])
    end
  else
    error('Only two dimensional tensors are supported now')
  end
  return self.rOutput
end

function Linear:updateRGradInput(input, rInput, gradOutput, rGradOutput)
  if input:dim() == 1 then
    self.rGradInput:resizeAs(input)
    self.rGradInput:mv(self.rWeight:t(), gradOutput)
    self.rGradInput:addmv(self.weight:t(), rGradOutput)
  elseif  input:dim() == 2 then
    self.rGradInput:resizeAs(input)
    self.rGradInput:mm(gradOutput, self.rWeight )
    self.rGradInput:addmm( rGradOutput, self.weight)
  else
    error('Only two dimensional tensors are supported now')
  end
  return self.rGradInput
end

function Linear:accRGradParameters(input, rInput, gradOutput, rGradOutput)

  if input:dim() == 1 then
    self.rGradWeight:addr(rGradOutput, input)
    self.rGradWeight:add(torch.ger(gradOutput, rInput))
    self.rGradBias:add(rGradOutput)
  elseif  input:dim() == 2 then
    self.rGradWeight:add(torch.mm(rGradOutput:t(), input))
    self.rGradWeight:add(torch.mm(gradOutput:t(), rInput))
    self.rGradBias:add(torch.sum(rGradOutput,1))

--    self.rGradWeight:div(input:size(1)) --
--    self.rGradBias:div(input:size(1))
  else
    error('Only two dimensional tensors are supported now')
  end
end

-- The LogSoftMax module
local LogSoftMax = nn.LogSoftMax
function LogSoftMax:updateROutput(input, rInput)
  if input:dim() == 1 then
    self.rOutput:resizeAs(rInput)
    self.rOutput:copy(rInput)
    
    local input_exp = torch.exp(input)
    local input_exp_sum = input_exp:sum()
    local s = torch.cmul(input_exp, rInput):sum()
    self.rOutput:add(-s/input_exp_sum)
  elseif  input:dim() == 2 then
    self.rOutput:resizeAs(rInput)
    self.rOutput:copy(rInput)
    
    local input_exp = torch.exp(input)
    local input_exp_sum = input_exp:sum(2)
    local s = torch.cmul(input_exp, rInput):sum(2)
    for i = 1, input:size(1) do
      self.rOutput[i]:add(-s[i][1]/input_exp_sum[i][1])  
    end
     
  else
    error('Only two dimensional tensors are supported now')
  end
  return self.rOutput
end

-- explanations: https://cswhjiang.github.io/2015/10/13/Roperator/
function LogSoftMax:updateRGradInput(input, rInput, gradOutput, rGradOutput)
  if input:dim() == 1 then
    self.rGradInput:resizeAs(rGradOutput)
    self.rGradInput:copy(rGradOutput) -- part3
    
    local gradOutput_sum = gradOutput:sum() -- delta
    local rGradOutput_sum = rGradOutput:sum() -- R(delta)
    
    local input_exp = torch.exp(input)
    local input_exp_sum = input_exp:sum()
    local p = torch.div(input_exp, input_exp_sum)
    
    local weightedRInputSum = torch.cmul(p, rInput):sum() -- a number
    local part1 = -torch.cmul(p, rInput - weightedRInputSum)*gradOutput_sum
    local part2 = -p*rGradOutput_sum
    
    self.rGradInput:add(part1):add(part2)
  elseif  input:dim() == 2 then
    self.rGradInput:resizeAs(rGradOutput)
    self.rGradInput:copy(rGradOutput)
    
    local gradOutput_sum = gradOutput:sum(2)
    local rGradOutput_sum = rGradOutput:sum(2) 
    
    local input_exp = torch.exp(input)
    local input_exp_sum = input_exp:sum(2)
    local p = torch.Tensor(input_exp:size())
    for i = 1, p:size(1) do 
      p[i] = torch.div(input_exp[i], input_exp_sum[i][1])
    end

    local weightedRInputSum = torch.cmul(p, rInput):sum(2) -- a number
    local part1 = torch.Tensor(rGradOutput:size())
    local part2 = torch.Tensor(rGradOutput:size())    
    
    for i = 1, part1:size(1) do
      local temp = torch.add(rInput[i], -weightedRInputSum[i][1])
      part1[i] = -torch.cmul(p[i], temp)*gradOutput_sum[i][1]
    end
    for i = 1, part2:size(1) do
      part2[i] = -p[i]*rGradOutput_sum[i][1]
    end  
    
    self.rGradInput:add(part1):add(part2)   

  else
    error('Only two dimensional tensors are supported now')
  end
  return self.rGradInput
end


-- For simple nonlinearities y = f(x) the forward propogation is R(x) * df(x).
local function TransferUpdateROutput(self, input, rInput)
   self.grad = self:updateGradInput(input, torch.ones(input:size()))
   self.rOutput:resizeAs(input)
   self.rOutput:cmul(rInput, self.grad)
   return self.rOutput
end

-- The backward prop is R(dE/dy) * df(x) + dE/dy * R(df(x)) where E is the cost
-- function, and R(df(x)) is d2f(x) * R(x).
local function TransferUpdateRGradInput(self, rGrad, input, rInput,
                                        gradOutput, rGradOutput)
   rGrad:cmul(rInput)
   self.rGradInput:resizeAs(input)
   self.rGradInput:cmul(rGradOutput, self.grad)
   self.rGradInput:addcmul(gradOutput, rGrad)
   return self.rGradInput
end

-- The Tanh module
local Tanh = nn.Tanh

Tanh.updateROutput = TransferUpdateROutput

function Tanh:updateRGradInput(input, ...)
   local rGrad = torch.cmul(torch.tanh(input), self.grad) * -2
   return TransferUpdateRGradInput(self, rGrad, input, ...)
end

-- The Sigmoid module
local Sigmoid = nn.Sigmoid

Sigmoid.updateROutput = TransferUpdateROutput

function Sigmoid:updateRGradInput(input, ...)
   local sigmoid = function(x)
      return torch.cdiv(torch.ones(x:size()), torch.exp(-x):add(1))
   end
   local sigmoid_input = sigmoid(input)
   local rGrad = (sigmoid_input -
                  torch.pow(sigmoid_input, 2) * 3 +
                  torch.pow(sigmoid_input, 3) * 2)
   return TransferUpdateRGradInput(self, rGrad, input, ...)
end
