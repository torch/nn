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
  local is_batch = true
  if input:dim() == 1 then
    input = input:view(1,-1)
    is_batch = false
  end

  self.output:resizeAs(input)

  self.norm = self.norm or input.new()
  self.normp = self.normp or input.new()
  self.buffer = self.buffer or input.new()

  if self.p % 2 ~= 0 then
    self.buffer:abs(input):pow(self.p)
  else
    self.buffer:pow(input,self.p)
  end
  self.normp:sum(self.buffer,2):add(self.eps)
  self.norm:pow(self.normp,1/self.p)
  self.output:cdiv(input,self.norm:view(-1,1):expandAs(self.output))

  if not is_batch then
    self.output = self.output[1]
  end
  return self.output
end

function Normalize:updateGradInput(input, gradOutput)
  assert(input:dim() <= 2, 'only 1d layer supported')
  assert(gradOutput:dim() <= 2, 'only 1d layer supported')

  local is_batch = true
  if input:dim() == 1 then
    input = input:view(1,-1)
    is_batch = false
  end

  local n = input:size(1) -- batch size
  local d = input:size(2) -- dimensionality of vectors
  -- compute diagonal term
  self.eye = self.eye or torch.eye(d):typeAs(input):view(1,d,d)
  local eyeExpand = self.eye:expand(n,d,d)
  self.diag = self.diag or self.eye.new()
  self.diag:cmul(eyeExpand, self.normp:view(n,1,1):expand(n,d,d))
  -- compute cross term
  self.buffer:abs(input):pow(self.p-2):cmul(input)
  local b1 = self.buffer:view(n,d,1)
  local b2 = input:view(n,1,d)

  self.diag:baddbmm(-1,b1,b2)
  -- compute the local gradient of the Lp transformation
  self.buffer:cmul(self.normp,self.norm)
  self.diag:cdiv(self.buffer:view(n,1,1):expand(n,d,d))
  -- chain the gradient
  self.gradInput:resize(n,d,1)
  self.gradInput:bmm(self.diag, gradOutput:view(n,d,1))
  self.gradInput = self.gradInput:view(n,d)
  
  if not is_batch then
    self.gradInput = self.gradInput[1]
  end

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
