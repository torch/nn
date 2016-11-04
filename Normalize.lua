local Normalize, parent = torch.class('nn.Normalize', 'nn.Module')
Normalize.__version = 2

function Normalize:__init(p, dim, eps)
  parent.__init(self)
  assert(p,'p-norm not provided')
  assert(p > 0, p..'-norm not supported')
  self.p = p
  self.dim = dim or -1
  self.eps = eps or 1e-10
  assert(self.dim % 1 == 0, 'dimension should be an integer')
  assert(self.dim ~= 0, "dimension can't be 0")
end

function Normalize:updateOutput(input)
  assert(math.abs(self.dim) <= input:dim(), 
    'input has less dimensions than the normalization dimension')
  local dim = self.dim
  if dim < 0 then
    dim = input:dim() + dim + 1
  end

  self.norm = self.norm or input.new()
  self.buffer = self.buffer or input.new()

  if self.p == math.huge then
    -- specialization for the infinity norm
    if not self._indices then
      if torch.type(self.output) == 'torch.CudaTensor' then
        self._indices = torch.CudaLongTensor and torch.CudaLongTensor() or torch.CudaTensor()
      else
        self._indices = torch.LongTensor()
      end
    end

    self.buffer:abs(input)
    torch.max(self.norm, self._indices, self.buffer, dim)
    self.norm:add(self.eps)
  else
    self.normp = self.normp or input.new()
    if self.p % 2 ~= 0 then
      self.buffer:abs(input):pow(self.p)
    else
      self.buffer:pow(input,self.p)
    end
    self.normp:sum(self.buffer, dim):add(self.eps)
    self.norm:pow(self.normp,1/self.p)
  end
  self.output:cdiv(input, self.norm:expandAs(input))

  return self.output
end

function Normalize:updateGradInput(input, gradOutput)
  assert(math.abs(self.dim) <= input:dim(), 
    'input has less dimensions than the normalization dimension')
  local dim = self.dim
  if dim < 0 then
    dim = input:dim() + dim + 1
  end

  self.cross = self.cross or input.new()
  -- compute diagonal term with gradOutput
  self.gradInput:resizeAs(input)
  if self.p == math.huge then
    -- specialization for the inf case
    self.gradInput:cmul(self.norm:expandAs(gradOutput),gradOutput)
    self.buffer:resizeAs(input):zero()
    self.cross:resizeAs(self.norm)
    self.cross:gather(input,dim,self._indices)
    self.cross:cdiv(self.norm)
    self.buffer:scatter(dim,self._indices,self.cross)
  else
    self.gradInput:cmul(self.normp:expandAs(gradOutput), gradOutput)
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
  end
  -- compute cross term in two steps
  self.cross:resizeAs(self.norm)

  -- instead of having a huge temporary matrix (b1*b2),
  -- do the computations as b1*(b2*gradOutput). This avoids redundant
  -- computation and also a huge buffer of size n*d^2
  self.buffer2 = self.buffer2 or input.new() -- nxd
  self.buffer2:cmul(input, gradOutput)
  self.cross:sum(self.buffer2, dim)

  self.buffer:cmul(self.cross:expandAs(self.buffer))
  self.gradInput:add(-1, self.buffer)

  -- reuse cross buffer for normalization
  if self.p == math.huge then
    self.cross:cmul(self.norm,self.norm)
  else
    self.cross:cmul(self.normp,self.norm)
  end
  self.gradInput:cdiv(self.cross:expandAs(gradOutput))

  return self.gradInput
end

function Normalize:__tostring__()
  local s
  -- different prints if the norm is integer
  if self.p % 1 == 0 then
    s = '%s(%d,%d)'
  else
    s = '%s(%f,%d)'
  end
  return string.format(s,torch.type(self),self.p, self.dim)
end

function Normalize:type(type, tensorCache)
    self._indices = nil
    parent.type(self, type, tensorCache)
    return self
end

function Normalize:clearState()
   nn.utils.clear(self, {
      '_output',
      '_indices',
      '_gradInput',
      'buffer',
      'norm',
      'normp',
      'cross',
   })
   return parent.clearState(self)
end

function Normalize:read(file, version)
   parent.read(self, file)
   if version < 2 then
      -- version 1 only supported 1D tensors
      self.dim = -1
   end
end
