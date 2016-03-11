local VolumetricDropout, Parent = torch.class('nn.VolumetricDropout', 'nn.Module')

function VolumetricDropout:__init(p)
   Parent.__init(self)
   self.p = p or 0.5
   self.train = true
   self.noise = torch.Tensor()
end

function VolumetricDropout:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   if self.train then
      if input:dim() == 5 then
        self.noise:resize(input:size(1), input:size(2), 1, 1, 1)
      elseif input:dim() == 4 then
        self.noise:resize(input:size(1), 1, 1, 1)
      else
        error('Input must be 5D (nbatch, nfeat, t, h, w) or 4D (nfeat, t, h, w)')
      end
      self.noise:bernoulli(1-self.p)
      -- We expand the random dropouts to the entire feature map because the
      -- features are likely correlated accross the map and so the dropout
      -- should also be correlated.
      self.output:cmul(torch.expandAs(self.noise, input))
   else
      self.output:mul(1-self.p)
   end
   return self.output
end

function VolumetricDropout:updateGradInput(input, gradOutput)
   if self.train then
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
      self.gradInput:cmul(torch.expandAs(self.noise, input)) -- simply mask the gradients with the noise vector
   else
      error('backprop only defined while training')
   end
   return self.gradInput
end

function VolumetricDropout:setp(p)
   self.p = p
end

function VolumetricDropout:__tostring__()
  return string.format('%s(%f)', torch.type(self), self.p)
end

function VolumetricDropout:clearState()
  if self.noise then
    self.noise:set()
  end
  return Parent.clearState(self)
end
