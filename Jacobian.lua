nn.Jacobian = {}

function nn.Jacobian.backward (module, input, param, dparam)
   local doparam = 0
   if param then
      doparam = 1
   end
   param = param or input
   -- output deriv
   module:forward(input)
   local dout = module.output.new():resizeAs(module.output)
   -- 1D view
   local sdout = module.output.new(dout:storage(),1,dout:nElement())
   -- jacobian matrix to calculate
   local jacobian = torch.Tensor(param:nElement(),dout:nElement()):zero()

   for i=1,sdout:nElement() do
      dout:zero()
      sdout[i] = 1
      module:zeroGradParameters()
      local din = module:updateGradInput(input, dout)
      module:accGradParameters(input, dout)
      if doparam == 1 then
	 jacobian:select(2,i):copy(dparam)
      else
	 jacobian:select(2,i):copy(din)
      end
   end
   return jacobian
end

function nn.Jacobian.backwardUpdate (module, input, param)

   -- output deriv
   module:forward(input)
   local dout = module.output.new():resizeAs(module.output)
   -- 1D view
   local sdout = module.output.new(dout:storage(),1,dout:nElement())
   -- jacobian matrix to calculate
   local jacobian = torch.Tensor(param:nElement(),dout:nElement()):zero()

   -- original param
   local origparam = param:clone()

   for i=1,sdout:nElement() do
      param:copy(origparam)
      dout:zero()
      sdout[i] = 1
      local din = module:updateGradInput(input, dout)
      module:accUpdateGradParameters(input, dout, 1)
      jacobian:select(2,i):copy(param)
   end

   param:copy(origparam)

   return jacobian
end

function nn.Jacobian.forward(module, input, param)
   param = param or input
   -- perturbation amount
   local small = 1e-6
   -- 1D view of input
   --local tst = param:storage()
   local sin = param.new(param):resize(param:nElement())--param.new(tst,1,tst:size())
   -- jacobian matrix to calculate
   local jacobian = torch.Tensor():resize(param:nElement(),module:forward(input):nElement())
   
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

function nn.Jacobian.forwardUpdate(module, input, param)
   -- perturbation amount
   local small = 1e-6
   -- 1D view of input
   --local tst = param:storage()
   local sin =  param.new(param):resize(param:nElement())--param.new(tst,1,tst:size())
   -- jacobian matrix to calculate
   local jacobian = torch.Tensor():resize(param:nElement(),module:forward(input):nElement())
   
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
      jacobian:select(1,i):mul(-1)
      jacobian:select(1,i):add(sin[i])
   end
   return jacobian
end

function nn.Jacobian.testJacobian (module, input, minval, maxval)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval
   input:copy(torch.rand(input:nElement()):mul(inrange):add(minval))
   local jac_fprop = nn.Jacobian.forward(module,input)
   local jac_bprop = nn.Jacobian.backward(module,input)
   local error = jac_fprop-jac_bprop
   return error:abs():max()
end

function nn.Jacobian.testJacobianParameters (module, input, param, dparam, minval, maxval)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval
   input:copy(torch.rand(input:nElement()):mul(inrange):add(minval))
   param:copy(torch.rand(param:nElement()):mul(inrange):add(minval))
   local jac_bprop = nn.Jacobian.backward(module, input, param, dparam)
   local jac_fprop = nn.Jacobian.forward(module, input, param)
   local error = jac_fprop - jac_bprop
   return error:abs():max()
end

function nn.Jacobian.testJacobianUpdateParameters (module, input, param, minval, maxval)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval
   input:copy(torch.rand(input:nElement()):mul(inrange):add(minval))
   param:copy(torch.rand(param:nElement()):mul(inrange):add(minval))
   local params_bprop = nn.Jacobian.backwardUpdate(module, input, param)
   local params_fprop = nn.Jacobian.forwardUpdate(module, input, param)

   local error = params_fprop - params_bprop
   return error:abs():max()
end

function nn.Jacobian.testIO(module,input, minval, maxval)
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

function nn.Jacobian.testAllUpdate(module, input, weight, gradWeight)
   local gradOutput
   local lr = torch.uniform(0.1, 1)
   local errors = {}

   -- accGradParameters
   local maccgp = module:clone()
   local weightc = maccgp[weight]:clone()
   maccgp:forward(input)
   gradOutput = torch.rand(maccgp.output:size())
   maccgp:zeroGradParameters()
   maccgp:updateGradInput(input, gradOutput)
   maccgp:accGradParameters(input, gradOutput)
   maccgp:updateParameters(lr)
   errors["accGradParameters"] = (weightc-maccgp[gradWeight]*lr-maccgp[weight]):norm()
   
   -- accUpdateGradParameters
   local maccugp = module:clone()
   maccugp:forward(input)
   maccugp:updateGradInput(input, gradOutput)
   maccugp:accUpdateGradParameters(input, gradOutput, lr)
   errors["accUpdateGradParameters"] = (maccugp[weight]-maccgp[weight]):norm()

   -- shared, accGradParameters
   local macsh1 = module:clone()
   local macsh2 = module:clone()
   macsh2:share(macsh1, weight)
   macsh1:forward(input)
   macsh2:forward(input)
   macsh1:zeroGradParameters()
   macsh2:zeroGradParameters()
   macsh1:updateGradInput(input, gradOutput)
   macsh2:updateGradInput(input, gradOutput)
   macsh1:accGradParameters(input, gradOutput)
   macsh2:accGradParameters(input, gradOutput)
   macsh1:updateParameters(lr)
   macsh2:updateParameters(lr)
   local err = (weightc-maccgp[gradWeight]*(lr*2)-macsh1[weight]):norm()
   err = err + (weightc-maccgp[gradWeight]*(lr*2)-macsh2[weight]):norm()
   errors["accGradParameters [shared]"] = err
   
   -- shared, accUpdateGradParameters
   local macshu1 = module:clone()
   local macshu2 = module:clone()
   macshu2:share(macshu1, weight)
   macshu1:forward(input)
   macshu2:forward(input)
   macshu1:updateGradInput(input, gradOutput)
   macshu2:updateGradInput(input, gradOutput)
   macshu1:accUpdateGradParameters(input, gradOutput, lr)
   macshu2:accUpdateGradParameters(input, gradOutput, lr)
   local err = (weightc-maccgp[gradWeight]*(lr*2)-macshu1[weight]):norm()
   err = err + (weightc-maccgp[gradWeight]*(lr*2)-macshu2[weight]):norm()
   errors["accUpdateGradParameters [shared]"] = err

   return errors
end
