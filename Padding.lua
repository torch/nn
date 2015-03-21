local Padding, parent = torch.class('nn.Padding', 'nn.Module')

-- pad can be positive (right) negative (left)
function Padding:__init(dim, pad, nInputDim, value)
   self.dim = dim
   self.pad = pad
   self.nInputDim = nInputDim
   self.value = value or 0
   self.outputSize = torch.LongStorage()
   parent.__init(self)
end

function Padding:updateOutput(input)
   self.outputSize:resize(input:dim())
   self.outputSize:copy(input:size())
   local dim = self.dim 
   if self.nInputDim and input:dim() ~= self.nInputDim then
      dim = dim + 1
   end
   self.outputSize[dim] = self.outputSize[dim] + math.abs(self.pad)
   self.output:resize(self.outputSize)
   self.output:fill(self.value)
   local outputWindow
   if self.pad > 0 then
      outputWindow = self.output:narrow(dim, 1, input:size(dim)) 
   else
      outputWindow = self.output:narrow(dim, 1 - self.pad, input:size(dim))
   end
   outputWindow:copy(input)
   return self.output
end

function Padding:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   local dim = self.dim 
   if self.nInputDim and input:dim() ~= self.nInputDim then
      dim = dim + 1
   end
   local gradOutputWindow
   if self.pad > 0 then
      gradOutputWindow = gradOutput:narrow(dim, 1, input:size(dim)) 
   else
      gradOutputWindow = gradOutput:narrow(dim, 1 - self.pad, input:size(dim))
   end
   self.gradInput:copy(gradOutputWindow:copy(input))
   return self.gradInput
end
