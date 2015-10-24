local Normalize, parent = torch.class('nn.Normalize', 'nn.Module')

function Normalize:__init(p,eps)
  parent.__init(self)
  assert(p,'p-norm not provided')
  assert(p > 0, p..'-norm not supported')
  self.p = p
  self.eps = eps or 1e-10
end

function Normalize:updateOutput(input)
  assert(input:dim() <= 2, 'only 1d layer supported')
  local input_size = input:size()
  if input:dim() == 1 then
    input = input:view(1,-1)
  end

  self._output = self._output or input.new()
  self.norm = self.norm or input.new()
  self.normp = self.normp or input.new()
  self.buffer = self.buffer or input.new()

  self._output:resizeAs(input)

  if self.p % 2 ~= 0 then
    self.buffer:abs(input):pow(self.p)
  else
    self.buffer:pow(input,self.p)
  end
  self.normp:sum(self.buffer,2):add(self.eps)
  self.norm:pow(self.normp,1/self.p)
  self._output:cdiv(input, self.norm:view(-1,1):expandAs(input))

  self.output = self._output:view(input_size)
  return self.output
end

function Normalize:updateGradInput(input, gradOutput)
  assert(input:dim() <= 2, 'only 1d layer supported')
  assert(gradOutput:dim() <= 2, 'only 1d layer supported')

  local input_size = input:size()
  if input:dim() == 1 then
    input = input:view(1,-1)
  end

  local n = input:size(1) -- batch size
  local d = input:size(2) -- dimensionality of vectors

  self._gradInput = self._gradInput or input.new()
  -- compute diagonal term with gradOutput
  self._gradInput:resize(n,d,1)
  gradOutput = gradOutput:view(n,d,1)
  self._gradInput:cmul(self.normp:view(n,1,1):expand(n,d,1), gradOutput)

  -- small optimizations for different p
  -- buffer = input*|input|^(p-2)
  if self.p % 2 ~= 0 then
    -- for non-even p, need to add absolute value
    if self.p < 2 then
      -- add eps to avoid possible division by 0
      self.buffer:abs(input):add(self.eps):pow(self.p-2):cmul(input)
    else
      self.buffer:abs(input):pow(self.p-2):cmul(input)
    end
  elseif self.p == 2 then
    -- special case for p == 2, pow(x,0) = 1
    self.buffer:copy(input)
  else
    -- p is even and > 2, pow(x,p) is always positive
    self.buffer:pow(input,self.p-2):cmul(input)
  end

  -- compute cross term in two steps
  self.cross = self.cross or input.new()
  self.cross:resize(n,1,1)

  local b1 = self.buffer:view(n,d,1)
  local b2 = input:view(n,1,d)
  -- instead of having a huge temporary matrix (b1*b2),
  -- do the computations as b1*(b2*gradOutput). This avoids redundant
  -- computation and also a huge buffer of size n*d^2
  self.cross:bmm(b2, gradOutput)
  self._gradInput:baddbmm(-1, b1, self.cross)

  -- reuse cross buffer for normalization
  self.cross:cmul(self.normp, self.norm)
  self._gradInput:cdiv(self.cross:view(n,1,1):expand(n,d,1))

  self._gradInput = self._gradInput:view(n,d)
  
  self.gradInput = self._gradInput:view(input_size)
  return self.gradInput
end

function Normalize:__tostring__()
  local s
  -- different prints if the norm is integer
  if self.p % 1 == 0 then
    s = '%s(%d)'
  else
    s = '%s(%f)'
  end
  return string.format(s,torch.type(self),self.p)
end
