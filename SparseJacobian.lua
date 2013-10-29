nn.SparseJacobian = {}

function nn.SparseJacobian.backward (module, input, param, dparam)
   local doparam = 0
   if param then
      doparam = 1
   end
   
   -- output deriv
   module:forward(input)
   local dout = module.output.new():resizeAs(module.output)
   -- 1D view
   local sdout = module.output.new(dout:storage(), 1, dout:nElement())
   -- jacobian matrix to calculate
   local jacobian
   if doparam == 1 then
      jacobian = torch.Tensor(param:nElement(), dout:nElement()):zero()
   else
      jacobian = torch.Tensor(input:size(1), dout:nElement()):zero()
   end

   for i=1,sdout:nElement() do
      dout:zero()
      sdout[i] = 1
      module:zeroGradParameters()
      local din = module:updateGradInput(input, dout)
      module:accGradParameters(input, dout)
      if doparam == 1 then
         jacobian:select(2,i):copy(dparam)
      else
         jacobian:select(2,i):copy(din:select(2,2))
      end
   end

   return jacobian
end

function nn.SparseJacobian.forward(module, input, param)
   local doparam = 0
   if param then
      doparam = 1
   end
   param = param or input

   -- perturbation amount
   local small = 1e-6
   -- 1D view of input
   --local tst = param:storage()
   local sin 
   if doparam == 1 then
      sin = param.new(param):resize(param:nElement())
   else
      sin = input.new(input):select(2,2)
   end
   
   local out = module:forward(input)
   -- jacobian matrix to calculate
   local jacobian 
   if doparam == 1 then
      jacobian = torch.Tensor():resize(param:nElement(),
                                       out:nElement())
   else
      jacobian = torch.Tensor():resize(input:size(1),
                                       out:nElement())
   end

   local outa = torch.Tensor(jacobian:size(2))
   local outb = torch.Tensor(jacobian:size(2))
   
   for i=1,sin:nElement() do      
      sin[i] = sin[i] - small
      outa:copy(module:forward(input))
      sin[i] = sin[i] + 2*small
      outb:copy(module:forward(input))
      sin[i] = sin[i] - small

      outb:add(-1,outa):div(2*small)
      jacobian:select(1,i):copy(outb)
   end

   return jacobian
end

function nn.SparseJacobian.testJacobian (module, input, minval, maxval)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval
   input:select(2,2):copy(torch.rand(input:size(1)):mul(inrange):add(minval))
   local jac_fprop = nn.SparseJacobian.forward(module,input)
   local jac_bprop = nn.SparseJacobian.backward(module,input)
   local error = jac_fprop-jac_bprop
   return error:abs():max()
end

function nn.SparseJacobian.testJacobianParameters (module, input, param, dparam, minval, maxval)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval
   input:select(2,2):copy(torch.rand(input:size(1)):mul(inrange):add(minval))
   param:copy(torch.rand(param:nElement()):mul(inrange):add(minval))
   local jac_bprop = nn.SparseJacobian.backward(module, input, param, dparam)
   local jac_fprop = nn.SparseJacobian.forward(module, input, param)
   local error = jac_fprop - jac_bprop
   return error:abs():max()
end

function nn.SparseJacobian.testIO(module,input, minval, maxval)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval

   -- run module
   module:forward(input)
   local go = module.output:clone():copy(torch.rand(module.output:nElement()):mul(inrange):add(minval))
   module:zeroGradParameters()
   module:updateGradInput(input,go)
   module:accGradParameters(input,go)

   local fo = module.output:clone()
   local bo = module.gradInput:clone()

   -- write module
   local f = torch.DiskFile('tmp.bin','w'):binary()
   f:writeObject(module)
   f:close()
   -- read module
   local m = torch.DiskFile('tmp.bin'):binary():readObject()
   m:forward(input)
   m:zeroGradParameters()
   m:updateGradInput(input,go)
   m:accGradParameters(input,go)
   -- cleanup
   os.remove('tmp.bin')

   local fo2 = m.output:clone()
   local bo2 = m.gradInput:clone()

   local errf = fo - fo2
   local errb = bo - bo2
   return errf:abs():max(), errb:abs():max()
end
