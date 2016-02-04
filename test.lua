-- you can easily test specific units like this:
-- th -lnn -e "nn.test{'LookupTable'}"
-- th -lnn -e "nn.test{'LookupTable', 'Add'}"

local mytester = torch.Tester()
local jac
local sjac

local precision = 1e-5
local expprecision = 1e-4

local nntest = {}

local function equal(t1, t2, msg)
   if (torch.type(t1) == "table") then
      for k, v in pairs(t2) do
         equal(t1[k], t2[k], msg)
      end
   else
      mytester:assertTensorEq(t1, t2, 0.00001, msg)
   end
end


--[[ Generate tests to exercise the tostring component of modules. ]]
local tostringTestModules = {
    nnLinear = nn.Linear(1, 2),
    nnReshape = nn.Reshape(10),
    nnSpatialZeroPadding = nn.SpatialZeroPadding(1, 1, 1, 1)}
for test_name, component in pairs(tostringTestModules) do
  nntest['tostring' .. test_name] =
    function ()
      mytester:assert(tostring(component):find(torch.type(component) .. '(',
                                                 1, true),
                      'nn components should have a descriptive tostring' ..
                      ' beginning with the classname')
    end
end


function nntest.Add()
   local inj_vals = {math.random(3,5), 1}  -- Also test the inj = 1 spatial case
   local ini = math.random(3,5)
   local ink = math.random(3,5)

   for ind, inj in pairs(inj_vals) do
      local input = torch.Tensor(ini,inj,ink):zero()
      local module = nn.Add(ini,inj,ink)

      -- 1D
      local err = jac.testJacobian(module,input)
      mytester:assertlt(err,precision, 'error on state ')

      local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
      mytester:assertlt(err,precision, 'error on bias ')

      local err = jac.testJacobianUpdateParameters(module, input, module.bias)
      mytester:assertlt(err,precision, 'error on bias [direct update] ')

      for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
         mytester:assertlt(err, precision, string.format('error on bias [%s]', t))
      end

      -- 2D
      local nframe = math.random(50,70)
      local input = torch.Tensor(nframe, ini,inj,ink):zero()

      local err = jac.testJacobian(module,input)
      mytester:assertlt(err,precision, 'error on state ')

      local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
      mytester:assertlt(err,precision, 'error on bias ')

      local err = jac.testJacobianUpdateParameters(module, input, module.bias)
      mytester:assertlt(err,precision, 'error on bias [direct update] ')

      for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
         mytester:assertlt(err, precision, string.format('error on bias [%s]', t))
      end

      -- IO
      local ferr,berr = jac.testIO(module,input)
      mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
      mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
   end
end

function nntest.CMul()
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.CMul(ini, inj, ink)

   -- 1D
   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err,precision, 'error on weight [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   -- 2D
   local nframe = math.random(50,70)
   local nframe = 5
   local input = torch.randn(nframe, ini,inj,ink)
   local output = module:forward(input)
   local output2 = torch.cmul(input, module.weight:view(1,ini,inj,ink):expandAs(input))
   mytester:assertTensorEq(output2, output, 0.000001, 'CMul forward 2D err')

   module:zeroGradParameters()
   local gradWeight = module.gradWeight:clone()
   local gradInput = module:backward(input, output)
   local gradInput2 = gradInput:clone():zero()
   local outputView = output:view(input:size(1), -1)
   gradInput2:view(input:size(1), -1):addcmul(1, module.weight:view(1,-1):expandAs(outputView), outputView)
   mytester:assertTensorEq(gradInput2, gradInput, 0.000001, 'CMul updateGradInput 2D err')
   mytester:assert(gradInput:isSameSizeAs(input), 'CMul gradInput 2D size err')

   local inputView = input:view(nframe, -1)
   local gradWeightView = gradWeight:view(1, -1)
   for i=1,nframe do
      gradWeightView:addcmul(1, inputView[i], outputView[i])
   end
   mytester:assertTensorEq(gradWeight, module.gradWeight, 0.000001, 'CMul accGradParameters 2D err')
   mytester:assert(module.weight:isSameSizeAs(module.gradWeight), 'CMul gradWeight size err')

   input:zero()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err,precision, 'error on weight [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format('error on weight [%s]', t))
   end
   
   -- Non-contiguous input or gradOutput
   local testModule = nn.CMul(4, 3, 5)
   local testInput = torch.rand(10, 3, 5):resize(10, 1, 3, 5):expand(10, 4, 3, 5)
   local testOutput = testModule:forward(testInput)

   mytester:assert(testOutput:isSameSizeAs(testInput), 'CMul non-contiguous forward err')

   local testGradOutput = torch.rand(10, 3, 5):resize(10, 1, 3, 5):expand(10, 4, 3, 5)
   testOutput = testModule:forward(testInput)
   local testGradInput = testModule:backward(testOutput, testGradOutput)

   mytester:assert(testGradInput:isSameSizeAs(testGradOutput), 'CMul non-contiguous backward err')
    
   -- IO
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Dropout()
   local p = 0.2 --prob of droping out a neuron
   local input = torch.Tensor(1000):fill((1-p))
   local module = nn.Dropout(p)
   -- version 2
   local output = module:forward(input)
   mytester:assert(math.abs(output:mean() - (1-p)) < 0.05, 'dropout output')
   local gradInput = module:backward(input, input)
   mytester:assert(math.abs(gradInput:mean() - (1-p)) < 0.05, 'dropout gradInput')
   -- test inplace version
   local module = nn.Dropout(p,nil,true)
   local output = module:forward(input:clone())
   mytester:assert(math.abs(output:mean() - (1-p)) < 0.05, 'dropout output')
   local gradInput = module:backward(input:clone(), input:clone())
   mytester:assert(math.abs(gradInput:mean() - (1-p)) < 0.05, 'dropout gradInput')

   -- version 1 (old nnx version)
   local input = input:fill(1)
   local module = nn.Dropout(p,true)
   local output = module:forward(input)
   mytester:assert(math.abs(output:mean() - (1-p)) < 0.05, 'dropout output')
   local gradInput = module:backward(input, input)
   mytester:assert(math.abs(gradInput:mean() - (1-p)) < 0.05, 'dropout gradInput')
end

function nntest.SpatialDropout()
   local p = 0.2 --prob of dropiing out a neuron
   local w = math.random(1,5)
   local h = math.random(1,5)
   local nfeats = 1000
   local input = torch.Tensor(nfeats, w, h):fill(1)
   local module = nn.SpatialDropout(p)
   module.train = true
   local output = module:forward(input)
   mytester:assert(math.abs(output:mean() - (1-p)) < 0.05, 'dropout output')
   local gradInput = module:backward(input, input)
   mytester:assert(math.abs(gradInput:mean() - (1-p)) < 0.05, 'dropout gradInput')
end

function nntest.SpatialDropoutBatch()
   local p = 0.2 --prob of dropiing out a neuron
   local bsz = math.random(1,5)
   local w = math.random(1,5)
   local h = math.random(1,5)
   local nfeats = 1000
   local input = torch.Tensor(bsz, nfeats, w, h):fill(1)
   local module = nn.SpatialDropout(p)
   module.train = true
   local output = module:forward(input)
   mytester:assert(math.abs(output:mean() - (1-p)) < 0.05, 'dropout output')
   local gradInput = module:backward(input, input)
   mytester:assert(math.abs(gradInput:mean() - (1-p)) < 0.05, 'dropout gradInput')
end

function nntest.ReLU()
   local input = torch.randn(3,4)
   local gradOutput = torch.randn(3,4)
   local module = nn.ReLU()
   local output = module:forward(input)
   local output2 = input:clone():gt(input, 0):cmul(input)
   mytester:assertTensorEq(output, output2, 0.000001, 'ReLU output')
   local gradInput = module:backward(input, gradOutput)
   local gradInput2 = input:clone():gt(input, 0):cmul(gradOutput)
   mytester:assertTensorEq(gradInput, gradInput2, 0.000001, 'ReLU gradInput')
end

function nntest.Exp()
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.Exp()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Log()
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.Log()

   local err = jac.testJacobian(module,input, 0.1, 10)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input, 0.1, 10)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.HardTanh()
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.HardTanh()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision ,  'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Clamp()
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local max_value =  math.abs(math.random())
   local min_value = -math.abs(math.random())
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.Clamp(min_value, max_value)

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision ,  'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Abs()
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.Abs()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision ,  'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Threshold()
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.Threshold(torch.uniform(-2,2),torch.uniform(-2,2))

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.ELU()
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.ELU(0.3)

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision ,  'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.PReLU()
   local ini = math.random(3,5)
   local input = torch.Tensor(ini):zero()

   local module = nn.PReLU(ini)

   -- 1D
   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err,precision, 'error on weight [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                        'error on weight [%s]', t))
   end

   -- 2D
   local nframe = math.random(1,7)
   local input = torch.Tensor(nframe, ini):zero()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err,precision, 'error on weight [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                        'error on weight [%s]', t))
   end

   -- 4D
   local nframe = math.random(1,7)
   local kW, kH = math.random(1,8), math.random(1,8)
   local input = torch.Tensor(nframe, ini, kW, kH):zero()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err,precision, 'error on weight [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                        'error on weight [%s]', t))
   end

   -- IO
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.RReLU()
   local nframe = math.random(1,7)
   local size = math.random(1,7)
   local kW, kH = math.random(1,8), math.random(1,8)
   local input = torch.Tensor(nframe, size, kW, kH):zero()

   local l = 1/math.random(5,8)
   local u = 1/math.random(3,5)

   -- test in evaluation mode (not inplace), RReLU behaves like LeakyReLU
   local module = nn.RReLU(l, u, false)
   mytester:assert(module.train, 'default mode ')
   module:evaluate()

   -- gradient check
   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   -- IO
   local ferr,berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

   -- test training and evalation mode
   for _,train in ipairs({true,false}) do
      -- test with separate output buffer and inplace
      for _,inplace in ipairs({false,true}) do
         module = nn.RReLU(l, u, inplace)
         if train then
            module:training()
         else
            module:evaluate()
         end
         input = torch.rand(nframe, size, kW, kH) - 0.5
         input:storage()[1] = -1
         local original_input = input:clone()
         local output = module:forward(input)
         mytester:assert(output:sign():eq(original_input:sign()):all(), 'sign flipped forward ')
         local gradOutput = torch.ones(output:size())
         local gradInput = module:backward(input, gradOutput)
         mytester:assert(gradInput:gt(0):eq(input:ne(0)):all(), 'gradient ')
         mytester:assert(gradInput:lt(1):eq(input:le(0)):all(), 'backward negative inputs ')
         mytester:assert(gradInput:eq(1):eq(input:gt(0)):all(), 'backward positive inputs ')
         if not train then
            local err = gradInput[input:le(0)]:mean()-(module.lower+module.upper)/2
            mytester:assertlt(err, precision, 'error on gradient ')
         end

         input = -torch.rand(1000)
         module:forward(input) -- fill internal noise tensor
         local g = module:backward(input, torch.ones(1000))
         local err = math.abs(g[input:le(0)]:mean()-(module.lower+module.upper)/2)
         mytester:assertlt(err, 0.05, 'mean deviation of gradient for negative inputs ')
      end
   end
end

function nntest.LeakyReLU()
   local input = torch.randn(3,4)
   local gradOutput = torch.randn(3,4)
   local negval = math.random()
   local module = nn.LeakyReLU(negval)
   local output = module:forward(input)
   local output2 = input:clone():gt(input, 0):cmul(input) + input:clone():le(input,0):cmul(input) * module.negval
   mytester:assertTensorEq(output, output2, 0.000001, 'LeakyReLU output')
   local gradInput = module:backward(input, gradOutput)
   local gradInput2 = input:clone():gt(input, 0):cmul(gradOutput) + input:clone():le(input,0):cmul(gradOutput) * module.negval
   mytester:assertTensorEq(gradInput, gradInput2, 0.000001, 'LeakyReLU gradInput')
end

function nntest.LeakyReLUIP()
   local input = torch.randn(3,4)
   local gradOutput = torch.randn(3,4)
   local negval = math.random()
   local module = nn.LeakyReLU(negval,true)
   local output = input:clone():gt(input, 0):cmul(input) + input:clone():le(input,0):cmul(input) * module.negval
   local output2 = module:forward(input)
   mytester:assertTensorEq(output2, output, 0.000001, 'LeakyReLU output')
   local gradInput = input:clone():gt(input, 0):cmul(gradOutput) + input:clone():le(input,0):cmul(gradOutput) * module.negval
   local gradInput2 = module:backward(input, gradOutput)
   mytester:assertTensorEq(gradInput2, gradInput, 0.000001, 'LeakyReLU gradInput')
end

function nntest.HardShrink()
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.HardShrink(math.random()/2)

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SoftShrink()
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.SoftShrink(math.random()/2)

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Power()
   local in1 = torch.rand(5,7)
   local module = nn.Power(2)
   local out = module:forward(in1)
   local err = out:dist(in1:cmul(in1))
   mytester:assertlt(err, 1e-15, torch.typename(module) .. ' - forward err ')

   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local pw = torch.uniform()*math.random(1,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.Power(pw)

   local err = nn.Jacobian.testJacobian(module, input, 0.1, 2)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module,input, 0.1, 2)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Normalize()
   -- compare forward against torch implementation
   -- and check gradient
   for _,p in pairs({1,2,3,4,1.5}) do
      local ini = math.random(3,10)
      local input = torch.randn(ini)
      local module = nn.Normalize(p)
      local out = module:forward(input)
      local expected = torch.div(input,input:norm(p))
      mytester:assertTensorEq(out, expected, 1e-7,
                              torch.typename(module) ..' (' .. p ..') - forward err ')

      local err = jac.testJacobian(module, input, -2, 2)
      mytester:assertlt(err, precision, 'error norm '..p..' on state ')
   end

   -- batch mode
   for _,p in pairs({1,2,3,4,torch.uniform()*math.random(1,10),math.huge}) do
      local ini = math.random(3,5)
      local inj = math.random(3,5)
      local ink = math.random(3,5)
      local input = torch.Tensor(inj, ini):zero()

      local module = nn.Normalize(p)

      local err = jac.testJacobian(module, input, -2, 2)
      mytester:assertlt(err, precision, 'error norm '..p..' on state ')
   end

   -- test IO correctness
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(inj, ini):zero()

   local module = nn.Normalize(2)

   local ferr, berr = jac.testIO(module,input, 0.1, 2)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

end

function nntest.Square()
   local in1 = torch.rand(5,7)
   local module = nn.Square()
   local out = module:forward(in1)
   local err = out:dist(in1:cmul(in1))
   mytester:assertlt(err, 1e-15, torch.typename(module) .. ' - forward err ')

   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.Square()

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Sqrt()
   local in1 = torch.rand(5,7)
   local module = nn.Sqrt()
   local out = module:forward(in1)
   local err = out:dist(in1:sqrt())
   mytester:assertlt(err, 1e-15, torch.typename(module) .. ' - forward err ')

   -- Test zero inputs; we will avoid a div-by-zero by setting to zero
   local zin = torch.DoubleTensor(5, 7):zero()
   module:forward(zin)
   local zgradout = torch.rand(5, 7)
   local zgradin = module:backward(zin, zgradout)
   mytester:assertTensorEq(zgradin, torch.DoubleTensor(5, 7):zero(), 0.000001, "error in sqrt backward singularity")

   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.Sqrt()

   local err = nn.Jacobian.testJacobian(module, input, 0.1, 2)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input, 0, 2)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Linear()
   local ini = math.random(3,5)
   local inj_vals = {math.random(3,5), 1}  -- Also test the inj = 1 spatial case
   local input = torch.Tensor(ini):zero()

   for ind, inj in pairs(inj_vals) do
     local module = nn.Linear(ini,inj)

     -- 1D
     local err = jac.testJacobian(module,input)
     mytester:assertlt(err,precision, 'error on state ')

     local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
     mytester:assertlt(err,precision, 'error on weight ')

     local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
     mytester:assertlt(err,precision, 'error on bias ')

     local err = jac.testJacobianUpdateParameters(module, input, module.weight)
     mytester:assertlt(err,precision, 'error on weight [direct update] ')

     local err = jac.testJacobianUpdateParameters(module, input, module.bias)
     mytester:assertlt(err,precision, 'error on bias [direct update] ')

     nn.hessian.enable()

     local err = jac.testDiagHessianInput(module, input)
     mytester:assertlt(err , precision, 'error on diagHessianInput')

     local err = jac.testDiagHessianWeight(module, input)
     mytester:assertlt(err , precision, 'error on diagHessianWeight')

     local err = jac.testDiagHessianBias(module, input)
     mytester:assertlt(err , precision, 'error on diagHessianBias')

     for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
        mytester:assertlt(err, precision, string.format(
                           'error on weight [%s]', t))
     end

     for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
        mytester:assertlt(err, precision, string.format(
                           'error on bias [%s]', t))
     end

     -- 2D
     local nframe = math.random(50,70)
     local input = torch.Tensor(nframe, ini):zero()

     local err = jac.testJacobian(module,input)
     mytester:assertlt(err,precision, 'error on state ')

     local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
     mytester:assertlt(err,precision, 'error on weight ')

     local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
     mytester:assertlt(err,precision, 'error on weight ')

     local err = jac.testJacobianUpdateParameters(module, input, module.weight)
     mytester:assertlt(err,precision, 'error on weight [direct update] ')

     local err = jac.testJacobianUpdateParameters(module, input, module.bias)
     mytester:assertlt(err,precision, 'error on bias [direct update] ')

     local err = jac.testDiagHessianInput(module, input)
     mytester:assertlt(err , precision, 'error on diagHessianInput')

     local err = jac.testDiagHessianWeight(module, input)
     mytester:assertlt(err , precision, 'error on diagHessianWeight')

     local err = jac.testDiagHessianBias(module, input)
     mytester:assertlt(err , precision, 'error on diag HessianBias')

     for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
        mytester:assertlt(err, precision, string.format(
                           'error on weight [%s]', t))
     end

     for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
        mytester:assertlt(err, precision, string.format(
                           'error on bias [%s]', t))
     end

     -- IO
     local ferr,berr = jac.testIO(module,input)
     mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
     mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
  end  -- for ind, inj in pairs(inj_vals) do
end

function nntest.SparseLinear()
   local ini = math.random(50,100)
   local inj = math.random(5,10)
   local numNonzero = math.random(3,5)

   local module = nn.SparseLinear(ini,inj)

   -- Create a random sparse vector
   local N = {}
   for i = 1, ini do N[i] = i end
   for i = 1, numNonzero do
      local j = math.random(i,ini)
      N[i], N[j] = N[j], N[i]
   end
   local input = torch.Tensor(numNonzero, 2):zero()
   for i = 1, numNonzero do input[{i,1}] = N[i] end
   local values = input:select(2,2)
   values:copy(torch.rand(values:nElement())):mul(2):add(-1)

   -- Check output
   local actual = module:forward(input)
   local expected = torch.Tensor(inj)
   for j = 1, inj do
      expected[j] = 0
      for i = 1,numNonzero do
         expected[j] = expected[j] + values[i] * module.weight[{j, N[i]}]
      end
   end
   local err = (expected - actual):abs():max()
   mytester:assertle(err, precision, 'error on result')

   -- Jacobian 1D
   local err = sjac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local err = sjac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

   local err = sjac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err,precision, 'error on bias ')

   local err = sjac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err,precision, 'error on weight [direct update] ')

   local err = sjac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err,precision, 'error on bias [direct update] ')

   for t,err in pairs(sjac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(sjac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'error on bias [%s]', t))
   end

   local ferr, berr = sjac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Bilinear()

   -- set up data:
   local N = 10
   local D1 = 5
   local D2 = 4
   local K  = 3
   local input  = {torch.randn(N, D1), torch.randn(N, D2)}
   local target = torch.randn(N, K)

   -- test forward-backward pass:
   local network = nn.Sequential():add(nn.Bilinear(D1, D2, K))
   local crit = nn.MSECriterion()
   crit:forward(network:forward(input), target)
   network:backward(input, crit:backward(network.output, target))

   -- function for gradient checking (nn.Jacobian does not work with tables):
   local function checkGradient(perturbation)

      -- prepare some variables:
      local perturbation = perturbation or 1e-6
      network:zeroGradParameters()
      local param, actual = network:getParameters()  -- flattened parameters

      -- loop over all to numerically approximate true Jacobian:
      local expected = param.new(param:nElement())
      for i = 1,param:nElement() do
         local orig = param[i]
         param[i] = orig - perturbation
         local outa = crit:forward(network:forward(input), target)
         param[i] = orig + perturbation
         local outb = crit:forward(network:forward(input), target)
         param[i] = orig
         expected[i] = (outb - outa) / (2 * perturbation)
      end

      -- compute Jacobian using the model:
      network:zeroGradParameters()
      crit:forward(network:forward(input), target)
      network:backward(input, crit:backward(network.output, target))

      -- compute error in Jacobian:
      local error = (actual - expected):abs():max()
      local expmax = expected:clone():abs():max()
      return ((error ~= 0) and (error / expmax) or 0), actual, expected
   end

   -- perform gradient check:
   mytester:assertlt(checkGradient(1e-9), 1e-4)
end

function nntest.Euclidean()
   local ini = math.random(5,7)
   local inj = math.random(5,7)
   local input = torch.randn(ini)
   local gradOutput = torch.randn(inj)
   local module = nn.Euclidean(ini,inj)
   local output = module:forward(input):clone()

   local output2 = torch.Tensor(inj):zero()
   for o = 1,module.weight:size(2) do
      output2[o] = input:dist(module.weight:select(2,o))
   end
   mytester:assertTensorEq(output, output2, 0.000001, 'Euclidean forward 1D err')

   local input2 = torch.randn(8, ini)
   input2[2]:copy(input)
   local output2 = module:forward(input2)
   mytester:assertTensorEq(output2[2], output, 0.000001, 'Euclidean forward 2D err')

   local output = module:forward(input):clone()
   module:zeroGradParameters()
   local gradInput = module:backward(input, gradOutput, 1):clone()
   local gradInput2 = torch.zeros(ini)
   local temp = input:clone()
   for o = 1,module.weight:size(2) do
      temp:copy(input)
      temp:add(-1,module.weight:select(2,o))
      temp:mul(gradOutput[o]/output[o])
      gradInput2:add(temp)
   end
   mytester:assertTensorEq(gradInput, gradInput2, 0.000001, 'Euclidean updateGradInput 1D err')

   local gradWeight = module.gradWeight:clone():zero()
   for o = 1,module.weight:size(2) do
      temp:copy(module.weight:select(2,o)):add(-1,input)
      temp:mul(gradOutput[o]/output[o])
      gradWeight:select(2,o):add(1, temp)
   end
   mytester:assertTensorEq(gradWeight, module.gradWeight, 0.000001, 'Euclidean accGradParameters 1D err')

   local input2 = input:view(1, -1):repeatTensor(8, 1)
   local gradOutput2 = gradOutput:view(1, -1):repeatTensor(8, 1)
   local output2 = module:forward(input2)
   module:zeroGradParameters()
   local gradInput2 = module:backward(input2, gradOutput2, 1/8)
   mytester:assertTensorEq(gradInput2[2], gradInput, 0.000001, 'Euclidean updateGradInput 2D err')

   mytester:assertTensorEq(gradWeight, module.gradWeight, 0.000001, 'Euclidean accGradParameters 2D err')

   input:zero()
   module.fastBackward = false
   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.WeightedEuclidean()
   local ini = math.random(5,7)
   local inj = math.random(5,7)
   local input = torch.randn(ini)
   local gradOutput = torch.randn(inj)
   local module = nn.WeightedEuclidean(ini,inj)

   local output = module:forward(input):clone()

   local output2 = torch.Tensor(inj):zero()
   local temp = input:clone()
   for o = 1,module.weight:size(2) do
      temp:copy(input):add(-1,module.weight:select(2,o))
      temp:cmul(temp)
      temp:cmul(module.diagCov:select(2,o)):cmul(module.diagCov:select(2,o))
      output2[o] = math.sqrt(temp:sum())
   end
   mytester:assertTensorEq(output, output2, 0.000001, 'WeightedEuclidean forward 1D err')

   local input2 = torch.randn(8, ini)
   input2[2]:copy(input)
   local output2 = module:forward(input2)
   mytester:assertTensorEq(output2[2], output, 0.000001, 'WeightedEuclidean forward 2D err')

   local output = module:forward(input):clone()
   module:zeroGradParameters()
   local gradInput = module:backward(input, gradOutput, 1):clone()
   local gradInput2 = torch.zeros(ini)
   for o = 1,module.weight:size(2) do
      temp:copy(input)
      temp:add(-1,module.weight:select(2,o))
      temp:cmul(module.diagCov:select(2,o)):cmul(module.diagCov:select(2,o))
      temp:mul(gradOutput[o]/output[o])
      gradInput2:add(temp)
   end
   mytester:assertTensorEq(gradInput, gradInput2, 0.000001, 'WeightedEuclidean updateGradInput 1D err')

   local gradWeight = module.gradWeight:clone():zero()
   local gradDiagCov = module.gradDiagCov:clone():zero()
   for o = 1,module.weight:size(2) do
      if output[o] ~= 0 then
         temp:copy(module.weight:select(2,o)):add(-1,input)
         temp:cmul(module.diagCov:select(2,o)):cmul(module.diagCov:select(2,o))
         temp:mul(gradOutput[o]/output[o])
         gradWeight:select(2,o):add(temp)

         temp:copy(module.weight:select(2,o)):add(-1,input)
         temp:cmul(temp)
         temp:cmul(module.diagCov:select(2,o))
         temp:mul(gradOutput[o]/output[o])
         gradDiagCov:select(2,o):add(temp)
      end
   end
   mytester:assertTensorEq(gradWeight, module.gradWeight, 0.000001, 'WeightedEuclidean accGradParameters gradWeight 1D err')
   mytester:assertTensorEq(gradDiagCov, module.gradDiagCov, 0.000001, 'WeightedEuclidean accGradParameters gradDiagCov 1D err')

   local input2 = input:view(1, -1):repeatTensor(8, 1)
   local gradOutput2 = gradOutput:view(1, -1):repeatTensor(8, 1)
   local output2 = module:forward(input2)
   module:zeroGradParameters()
   local gradInput2 = module:backward(input2, gradOutput2, 1/8)
   mytester:assertTensorEq(gradInput2[2], gradInput, 0.000001, 'WeightedEuclidean updateGradInput 2D err')

   mytester:assertTensorEq(gradWeight, module.gradWeight, 0.000001, 'WeightedEuclidean accGradParameters gradWeight 2D err')
   mytester:assertTensorEq(gradDiagCov, module.gradDiagCov, 0.000001, 'WeightedEuclidean accGradParameters gradDiagCov 2D err')

   input:zero()
   module.fastBackward = false

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input, module.diagCov, module.gradDiagCov)
   mytester:assertlt(err,precision, 'error on bias ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

   input:zero()
   module:zeroGradParameters()
   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input, module.diagCov, module.gradDiagCov)
   mytester:assertlt(err,precision, 'error on bias ')

   local ferr,berr = jac.testIO(module,input2)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

local function criterionJacobianTest1D(cri, input, target)
   local eps = 1e-6
   local _ = cri:forward(input, target)
   local dfdx = cri:backward(input, target)
   -- for each input perturbation, do central difference
   local centraldiff_dfdx = torch.Tensor():resizeAs(dfdx)
   local input_s = input:storage()
   local centraldiff_dfdx_s = centraldiff_dfdx:storage()
   for i=1,input:nElement() do
      -- f(xi + h)
      input_s[i] = input_s[i] + eps
      local fx1 = cri:forward(input, target)
      -- f(xi - h)
      input_s[i] = input_s[i] - 2*eps
      local fx2 = cri:forward(input, target)
      -- f'(xi) = (f(xi + h) - f(xi - h)) / 2h
      local cdfx = (fx1 - fx2) / (2*eps)
      -- store f' in appropriate place
      centraldiff_dfdx_s[i] = cdfx
      -- reset input[i]
      input_s[i] = input_s[i] + eps
   end

   -- compare centraldiff_dfdx with :backward()
   local err = (centraldiff_dfdx - dfdx):abs():max()
   mytester:assertlt(err, precision, 'error in difference between central difference and :backward')
end

local function criterionJacobianTest1DTable(cri, input0, target)
   -- supposes input is a tensor, which is splitted in the first dimension
   local input = input0:split(1,1)
   for i=1,#input do
      input[i] = input[i][1]
   end
   local eps = 1e-6
   local _ = cri:forward(input, target)
   local dfdx = cri:backward(input, target)
   -- for each input perturbation, do central difference
   local centraldiff_dfdx = torch.Tensor():resizeAs(input0)
   local input_s = input0:storage()
   local centraldiff_dfdx_s = centraldiff_dfdx:storage()
   for i=1,input0:nElement() do
      -- f(xi + h)
      input_s[i] = input_s[i] + eps
      local fx1 = cri:forward(input, target)
      -- f(xi - h)
      input_s[i] = input_s[i] - 2*eps
      local fx2 = cri:forward(input, target)
      -- f'(xi) = (f(xi + h) - f(xi - h)) / 2h
      local cdfx = (fx1 - fx2) / (2*eps)
      -- store f' in appropriate place
      centraldiff_dfdx_s[i] = cdfx
      -- reset input[i]
      input_s[i] = input_s[i] + eps
   end
   local centraldiff_dfdx_t = centraldiff_dfdx:split(1,1)
   for i=1,#centraldiff_dfdx_t do
      centraldiff_dfdx_t[i] = centraldiff_dfdx_t[i][1]
   end
   for i=1,#centraldiff_dfdx_t do
      -- compare centraldiff_dfdx with :backward()
      local err = (centraldiff_dfdx_t[i] - dfdx[i]):abs():max()
      mytester:assertlt(err, precision, 'error in difference between central difference and :backward')
   end
end

function nntest.SmoothL1Criterion()
   local input = torch.rand(10)
   local target = input:clone():add(torch.rand(10))
   local cri = nn.SmoothL1Criterion()
   criterionJacobianTest1D(cri, input, target)
end

function nntest.MSECriterion()
   local input = torch.rand(10)
   local target = input:clone():add(torch.rand(10))
   local cri = nn.MSECriterion()
   criterionJacobianTest1D(cri, input, target)
end

function nntest.MarginCriterion()
   local input = torch.rand(100)
   local target = input:clone():add(torch.rand(100))
   local cri = nn.MarginCriterion()
   criterionJacobianTest1D(cri, input, target)
end

function nntest.MultiMarginCriterion()
   local input = torch.rand(100)
   local target = math.random(1,100)
   local cri = nn.MultiMarginCriterion(math.random(1,2))
   criterionJacobianTest1D(cri, input, target)
end

function nntest.MarginRankingCriterion()
   local input = {torch.rand(1), torch.rand(1)}
   local mrc = nn.MarginRankingCriterion()
   local output = mrc:forward(input, 1)
   local gradInput = mrc:backward(input, 1)
   -- cast to float
   local input2 = {input[1]:float(), input[2]:float()}
   local mrc2 = mrc:clone():float()
   local output2 = mrc2:forward(input2, 1)
   local gradInput2 = mrc2:backward(input2, 1)
   mytester:assert(math.abs(output2 - output) < 0.00001, "MRC:type() forward error")
   mytester:assertTensorEq(gradInput[1]:float(), gradInput2[1], 0.00001, "MRC:type() backward error 1")
   mytester:assert(torch.type(gradInput2[1]) == 'torch.FloatTensor', "MRC:type() error 1")
   mytester:assertTensorEq(gradInput[2]:float(), gradInput2[2], 0.00001, "MRC:type() backward error 2")
   mytester:assert(torch.type(gradInput2[2]) == 'torch.FloatTensor', "MRC:type() error 2")

   -- batch, sizeAverage true, jacobian
   local margin = math.random()*2-1
   local batch_size = math.random(2,10)
   local crit = nn.MarginRankingCriterion(margin)
   crit.sizeAverage = true
   local v = torch.rand(2,batch_size)
   local t = torch.Tensor(batch_size):random(0,1):mul(2):add(-1)
   criterionJacobianTest1DTable(crit,v,t)

   -- batch, sizeAverage false, jacobian
   local margin = math.random()*2-1
   local crit = nn.MarginRankingCriterion(margin)
   crit.sizeAverage = false
   local v = torch.rand(2,batch_size)
   local t = torch.Tensor(batch_size):random(0,1):mul(2):add(-1)
   criterionJacobianTest1DTable(crit,v,t)

end

function nntest.ParallelCriterion()
   local input = {torch.rand(2,10), torch.randn(2,10)}
   local target = {torch.IntTensor{1,8}, torch.randn(2,10)}
   local nll = nn.ClassNLLCriterion()
   local mse = nn.MSECriterion()
   local pc = nn.ParallelCriterion():add(nll, 0.5):add(mse)
   local output = pc:forward(input, target)
   local output2 = nll:forward(input[1], target[1])/2 + mse:forward(input[2], target[2])
   mytester:assert(math.abs(output2 - output) < 0.00001, "ParallelCriterion forward error")
   local gradInput2 = {nll:backward(input[1], target[1]):clone():div(2), mse:backward(input[2], target[2])}
   local gradInput = pc:backward(input, target)
   mytester:assertTensorEq(gradInput[1], gradInput2[1], 0.000001, "ParallelCriterion backward error 1")
   mytester:assertTensorEq(gradInput[2], gradInput2[2], 0.000001, "ParallelCriterion backward error 2")

   -- test type
   pc:float()
   gradInput[1], gradInput[2] = gradInput[1]:clone(), gradInput[2]:clone()
   local input3 = {input[1]:float(), input[2]:float()}
   local target3 = {target[1]:float(), target[2]:float()}
   local output3 = pc:forward(input3, target3)
   local gradInput3 = pc:backward(input3, target3)
   mytester:assert(math.abs(output3 - output) < 0.00001, "ParallelCriterion forward error type")
   mytester:assertTensorEq(gradInput[1]:float(), gradInput3[1], 0.000001, "ParallelCriterion backward error 1 type")
   mytester:assertTensorEq(gradInput[2]:float(), gradInput3[2], 0.000001, "ParallelCriterion backward error 2 type")

   -- test repeatTarget
   local input = {torch.rand(2,10), torch.randn(2,10)}
   local target = torch.randn(2,10)
   local mse = nn.MSECriterion()
   local pc = nn.ParallelCriterion(true):add(mse, 0.5):add(mse:clone())
   local output = pc:forward(input, target)
   local output2 = mse:forward(input[1], target)/2 + mse:forward(input[2], target)
   mytester:assert(math.abs(output2 - output) < 0.00001, "ParallelCriterion repeatTarget forward error")
   local gradInput = pc:backward(input, target)
   local gradInput2 = {mse:backward(input[1], target):clone():div(2), mse:backward(input[2], target)}
   mytester:assertTensorEq(gradInput[1], gradInput2[1], 0.000001, "ParallelCriterion repeatTarget backward error 1")
   mytester:assertTensorEq(gradInput[2], gradInput2[2], 0.000001, "ParallelCriterion repeatTarget backward error 2")

   -- table input
   local input = {torch.randn(2,10), {torch.rand(2,10), torch.randn(2,10)}}
   local target = {torch.IntTensor{2,5}, {torch.IntTensor{1,8}, torch.randn(2,10)}}
   local nll2 = nn.ClassNLLCriterion()
   local nll = nn.ClassNLLCriterion()
   local mse = nn.MSECriterion()
   local pc = nn.ParallelCriterion():add(nll, 0.5):add(mse)
   local pc2 = nn.ParallelCriterion():add(nll2, 0.4):add(pc)
   local output = pc2:forward(input, target)
   local output2 = nll2:forward(input[1], target[1])*0.4 + nll:forward(input[2][1], target[2][1])/2 + mse:forward(input[2][2], target[2][2])
   mytester:assert(math.abs(output2 - output) < 0.00001, "ParallelCriterion table forward error")
   local gradInput2 = {
       nll2:backward(input[1], target[1]):clone():mul(0.4),
      {nll:backward(input[2][2], target[2][1]):clone():div(2), mse:backward(input[2][2], target[2][2])}
   }
   local gradInput = pc2:backward(input, target)
   mytester:assertTensorEq(gradInput[1], gradInput2[1], 0.000001, "ParallelCriterion table backward error 1")
   mytester:assertTensorEq(gradInput[2][1], gradInput2[2][1], 0.000001, "ParallelCriterion table backward error 2")
   mytester:assertTensorEq(gradInput[2][2], gradInput2[2][2], 0.000001, "ParallelCriterion table backward error 3")
end

function nntest.MultiCriterion()
   local input = torch.rand(2,10)
   local target = torch.IntTensor{1,8}
   local nll = nn.ClassNLLCriterion()
   local nll2 = nn.CrossEntropyCriterion()
   local mc = nn.MultiCriterion():add(nll, 0.5):add(nll2)
   local output = mc:forward(input, target)
   local output2 = nll:forward(input, target)/2 + nll2:forward(input, target)
   mytester:assert(math.abs(output2 - output) < 0.00001, "MultiCriterion forward error")
   local gradInput = mc:backward(input, target)
   local gradInput2 = nll:backward(input, target):clone():div(2):add(nll2:backward(input, target))
   mytester:assertTensorEq(gradInput, gradInput2, 0.000001, "MultiCriterion backward error ")

   -- test type
   mc:float()
   gradInput = gradInput:clone()
   local input3 = input:float()
   local target3 = target:float()
   local output3 = mc:forward(input3, target3)
   local gradInput3 = mc:backward(input3, target3)
   mytester:assert(math.abs(output3 - output) < 0.00001, "MultiCriterion forward error type")
   mytester:assertTensorEq(gradInput:float(), gradInput3, 0.000001, "MultiCriterion backward error type")

   -- test table input
   mc:double()
   local input = {torch.randn(2,10), {torch.randn(2,10), torch.randn(2,10)}}
   local target = {torch.IntTensor{1,8}, {torch.IntTensor{5,6}, torch.IntTensor{4,3}}}
   local pnllc = nn.ParallelCriterion():add(nll):add(nn.ParallelCriterion():add(nll:clone()):add(nll:clone()))
   local pnllc2 = nn.ParallelCriterion():add(nll2):add(nn.ParallelCriterion():add(nll2:clone()):add(nll2:clone()))
   local mc = nn.MultiCriterion():add(pnllc, 0.5):add(pnllc2)
   local output = mc:forward(input, target)
   local output2 = pnllc:forward(input, target)/2 + pnllc2:forward(input, target)
   mytester:assert(math.abs(output2 - output) < 0.00001, "MultiCriterion forward table error")
   local gradInput = mc:backward(input, target)
   local gradInput2 = pnllc:clone():backward(input, target)
   local gradInput2b = pnllc2:backward(input, target)
   gradInput2[1]:div(2):add(gradInput2b[1])
   gradInput2[2][1]:div(2):add(gradInput2b[2][1])
   gradInput2[2][2]:div(2):add(gradInput2b[2][2])
   mytester:assertTensorEq(gradInput[1], gradInput2[1], 0.000001, "MultiCriterion backward table 1 error ")
   mytester:assertTensorEq(gradInput[2][1], gradInput2[2][1], 0.000001, "MultiCriterion backward table 2 error ")
   mytester:assertTensorEq(gradInput[2][2], gradInput2[2][2], 0.000001, "MultiCriterion backward table 3 error ")
end

function nntest.WeightedMSECriterion()
   local input = torch.rand(10)
   local target = input:clone():add(torch.rand(10))
   local cri = nn.WeightedMSECriterion(torch.rand(10))
   criterionJacobianTest1D(cri, input, target)
end

function nntest.BCECriterion()
   local eps = 1e-2
   local input = torch.rand(10)*(1-eps) + eps/2
   local target = torch.rand(10)*(1-eps) + eps/2
   local cri = nn.BCECriterion()
   criterionJacobianTest1D(cri, input, target)
   --with weights
   local weights= torch.rand(10)*(1-eps) + eps/2
   local cri = nn.BCECriterion(weights)
   criterionJacobianTest1D(cri, input, target)
   -- with weights + batch
   local bsz = 5
   local input = torch.rand(bsz, 10)*(1-eps) + eps/2
   local target = torch.rand(bsz, 10)*(1-eps) + eps/2
   criterionJacobianTest1D(cri, input, target)
end

function nntest.DistKLDivCriterion()
   local input = torch.rand(10)
   local target = input:clone():add(torch.rand(10))
   local cri = nn.DistKLDivCriterion(true)  -- sizeAverage = true
   criterionJacobianTest1D(cri, input, target)
   cri = nn.DistKLDivCriterion(false)  -- sizeAverage = false
   criterionJacobianTest1D(cri, input, target)
end

function nntest.ClassNLLCriterion()
   local numLabels = math.random(5,10)
   local input = torch.rand(numLabels)
   local target = math.random(1,numLabels)

   -- default ClassNLLCriterion
   local cri = nn.ClassNLLCriterion()
   criterionJacobianTest1D(cri, input, target)

   -- ClassNLLCriterion with weights
   local weights = torch.rand(numLabels)
   weights = weights / weights:sum()
   cri = nn.ClassNLLCriterion(weights)
   criterionJacobianTest1D(cri, input, target)
end

function nntest.CrossEntropyCriterion()
   -- stochastic
   local numLabels = math.random(5, 10)
   local input = torch.zeros(numLabels)
   local target = torch.random(1, numLabels)

   local cri = nn.CrossEntropyCriterion()
   criterionJacobianTest1D(cri, input, target)

   -- batch
   local numLabels = math.random(5,10)
   local bsz = math.random(3, 7)
   local input = torch.zeros(bsz, numLabels)
   local target = torch.Tensor(bsz):random(1, numLabels)

   local cri = nn.CrossEntropyCriterion()
   criterionJacobianTest1D(cri, input, target)

   -- with weights
   local weights = torch.rand(numLabels)
   weights = weights / weights:sum()
   cri = nn.CrossEntropyCriterion(weights)
   criterionJacobianTest1D(cri, input, target)
end

function nntest.LogSigmoid()
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.LogSigmoid()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.LogSoftmax()
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local input = torch.Tensor(ini,inj):zero()
   local module = nn.LogSoftMax()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err, 1e-3, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

-- function nntest.TemporalLogSoftmax()
--    local ini = math.random(10,20)
--    local inj = math.random(10,20)
--    local input = torch.Tensor(ini,inj):zero()
--    local module = nn.TemporalLogSoftMax()

--    local err = jac.testJacobian(module,input)
--    mytester:assertlt(err,precision, 'error on state ')

--    local ferr,berr = jac.testIO(module,input)
--    mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
--    mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
-- end

function nntest.Max()
   -- 1D
   local ini = math.random(3,7)
   local input = torch.Tensor(ini):zero()
   local module = nn.Max(1)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   -- negative dimension
   local module = nn.Max(-1)
   local input = torch.Tensor({1, 2, 3})
   local expected = torch.Tensor({3})
   local output = module:forward(input)
   mytester:assertlt(torch.norm(output-expected), precision, 'error on forward ')
   -- batch
   local module = nn.Max(1, 1)
   local input = torch.Tensor({{1, 2, 3},{4, 5, 6}})
   local expected = torch.Tensor({3, 6})
   local output = module:forward(input)
   mytester:assertlt(torch.norm(output-expected), precision, 'error on forward ')

   -- 3D
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ini,inj*ink):zero()
   local module = nn.Max(1)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Min()
   -- 1D
   local ini = math.random(3,7)
   local input = torch.Tensor(ini):zero()
   local module = nn.Min(1)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   -- negative dimension
   local module = nn.Min(-1)
   local input = torch.Tensor({1, 2, 3})
   local expected = torch.Tensor({1})
   local output = module:forward(input)
   mytester:assertlt(torch.norm(output-expected), precision, 'error on forward ')
   -- batch
   local module = nn.Min(1, 1)
   local input = torch.Tensor({{1, 2, 3},{4, 5, 6}})
   local expected = torch.Tensor({1, 4})
   local output = module:forward(input)
   mytester:assertlt(torch.norm(output-expected), precision, 'error on forward ')

   -- 3D
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ini,inj*ink):zero()
   local module = nn.Min(1)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Mean()
   -- 1D
   local ini = math.random(3,7)
   local input = torch.Tensor(ini):zero()
   local module = nn.Mean(1)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   -- negative dimension
   local module = nn.Mean(-1)
   local input = torch.Tensor({1, 2, 3})
   local expected = torch.Tensor({2})
   local output = module:forward(input)
   mytester:assertlt(torch.norm(output-expected), precision, 'error on forward ')
   -- batch
   local module = nn.Mean(1, 1)
   local input = torch.Tensor({{1, 2, 3},{4, 5, 6}})
   local expected = torch.Tensor({2, 5})
   local output = module:forward(input)
   mytester:assertlt(torch.norm(output-expected), precision, 'error on forward ')

   -- 3D
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.Mean(torch.random(1,3))

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Mul()
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.Mul()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')
   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')
   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err,precision, 'error on weight [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Sigmoid()
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.Sigmoid()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Softmax()
   local ini = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ink, ini):zero()
   local module = nn.SoftMax()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,expprecision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SpatialSoftMax()
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local inl = math.random(3,5)
   local input = torch.Tensor(inl, ink, inj, ini):zero()
   local module = nn.SpatialSoftMax()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,expprecision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Softmin()
   local ini = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ink, ini):zero()
   local module = nn.SoftMin()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,expprecision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Softsign()
   local ini = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ink, ini):zero()
   local module = nn.SoftSign()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SoftPlus()
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.SoftPlus()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SpatialSubtractiveNormalization_2dkernel()
   local inputSize = math.random(6,9)
   local kersize = 3
   local nbfeatures = math.random(3,5)
   local kernel = torch.Tensor(kersize,kersize):fill(1)
   local module = nn.SpatialSubtractiveNormalization(nbfeatures,kernel)
   local input = torch.rand(nbfeatures,inputSize,inputSize/2)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

    -- test batch mode
   local output = module:forward(input):clone()
   local gradOutput = output:clone():uniform(0,1)
   local gradInput = module:backward(input, gradOutput):clone()
   local batchSize = 4
   local input2 = torch.rand(batchSize,nbfeatures,inputSize,inputSize/2)
   input2[2]:copy(input)

   local output2 = module:forward(input2)
   local gradOutput2 = output2:clone():uniform(0,1)
   gradOutput2[2]:copy(gradOutput)
   local gradInput2 = module:backward(input2, gradOutput2)

   mytester:assertTensorEq(output2[2], output, 0.000001, "SpatialSubstractiveNormalization 2d forward batch err")
   mytester:assertTensorEq(gradOutput2[2], gradOutput, 0.000001, "SpatialSubstractiveNormalization 2d backward batch err")

   local err = jac.testJacobian(module,input2)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input2)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

end

function nntest.SpatialSubtractiveNormalization_1dkernel()
   local inputSize = math.random(6,9)
   local kersize = 3
   local nbfeatures = math.random(3,5)
   local kernel = torch.Tensor(kersize):fill(1)
   local module = nn.SpatialSubtractiveNormalization(nbfeatures,kernel)
   local input = torch.rand(nbfeatures,inputSize,inputSize/2)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

    -- test batch mode
   local output = module:forward(input):clone()
   local gradOutput = output:clone():uniform(0,1)
   local gradInput = module:backward(input, gradOutput):clone()
   local batchSize = 4
   local input2 = torch.rand(batchSize,nbfeatures,inputSize,inputSize/2)
   input2[2]:copy(input)

   local output2 = module:forward(input2)
   local gradOutput2 = output2:clone():uniform(0,1)
   gradOutput2[2]:copy(gradOutput)
   local gradInput2 = module:backward(input2, gradOutput2)

   mytester:assertTensorEq(output2[2], output, 0.000001, "SpatialSubstractiveNormalization 1d forward batch err")
   mytester:assertTensorEq(gradOutput2[2], gradOutput, 0.000001, "SpatialSubstractiveNormalization 1d backward batch err")

   local err = jac.testJacobian(module,input2)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input2)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SpatialDivisiveNormalization_2dkernel()
   local inputSize = math.random(6,9)
   local kersize = 3
   local nbfeatures = math.random(3,5)
   local kernel = torch.Tensor(kersize,kersize):fill(1)
   local module = nn.SpatialDivisiveNormalization(nbfeatures,kernel)
   local input = torch.rand(nbfeatures,inputSize,inputSize/2)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

   -- test batch mode
   local output = module:forward(input):clone()
   local gradOutput = output:clone():uniform(0,1)
   local gradInput = module:backward(input, gradOutput):clone()
   local batchSize = 4
   local input2 = torch.rand(batchSize,nbfeatures,inputSize,inputSize/2)
   input2[2]:copy(input)

   local output2 = module:forward(input2)
   local gradOutput2 = output2:clone():uniform(0,1)
   gradOutput2[2]:copy(gradOutput)
   local gradInput2 = module:backward(input2, gradOutput2)

   mytester:assertTensorEq(output2[2], output, 0.000001, "SpatialDivisiveNormalization 2d forward batch err")
   mytester:assertTensorEq(gradOutput2[2], gradOutput, 0.000001, "SpatialDivisiveNormalization 2d backward batch err")

   local err = jac.testJacobian(module,input2)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input2)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SpatialDivisiveNormalization_1dkernel()
   local inputSize = math.random(6,9)
   local kersize = 3
   local nbfeatures = math.random(3,5)
   local kernel = torch.Tensor(kersize):fill(1)
   local module = nn.SpatialDivisiveNormalization(nbfeatures,kernel)
   local input = torch.rand(nbfeatures,inputSize,inputSize/2)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

    -- test batch mode
   local output = module:forward(input):clone()
   local gradOutput = output:clone():uniform(0,1)
   local gradInput = module:backward(input, gradOutput):clone()
   local batchSize = 4
   local input2 = torch.rand(batchSize,nbfeatures,inputSize,inputSize/2)
   input2[2]:copy(input)

   local output2 = module:forward(input2)
   local gradOutput2 = output2:clone():uniform(0,1)
   gradOutput2[2]:copy(gradOutput)
   local gradInput2 = module:backward(input2, gradOutput2)

   mytester:assertTensorEq(output2[2], output, 0.000001, "SpatialDivisiveNormalization 1d forward batch err")
   mytester:assertTensorEq(gradOutput2[2], gradOutput, 0.000001, "SpatialDivisiveNormalization 1d backward batch err")

   local err = jac.testJacobian(module,input2)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input2)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SpatialContrastiveNormalization()
   local inputSize = math.random(6,9)
   local kersize = 3
   local nbfeatures = math.random(3,5)
   local kernel = torch.Tensor(kersize,kersize):fill(1)
   local module = nn.SpatialContrastiveNormalization(nbfeatures,kernel)
   local input = torch.rand(nbfeatures,inputSize,inputSize/2)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

   -- test batch mode and type
   local output = module:forward(input):clone()
   local gradOutput = output:clone():uniform(0,1)
   local gradInput = module:backward(input, gradOutput):clone()
   local batchSize = 4
   local input2 = torch.rand(batchSize,nbfeatures,inputSize,inputSize/2):float()
   input2[2]:copy(input)

   module:float() -- type-cast
   local output2 = module:forward(input2)
   local gradOutput2 = output2:clone():uniform(0,1)
   gradOutput2[2]:copy(gradOutput)
   local gradInput2 = module:backward(input2, gradOutput2)

   mytester:assertTensorEq(output2[2], output:float(), 0.000002, "SpatialContrastiveNormalization 2d forward batch err")
   mytester:assertTensorEq(gradOutput2[2], gradOutput:float(), 0.000002, "SpatialContrastiveNormalization 2d backward batch err")

   module:double()
   input2 = input2:double()
   local err = jac.testJacobian(module,input2)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input2)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SpatialCrossMapLRN()
   local inputSize = math.random(6,9)
   local size = math.random(1,3)*2+1
   local nbfeatures = math.random(3,8)
   local alpha = math.random(1,100)/100
   local beta  = math.random(0,100)/100
   local k = math.random(1,3)
   local module = nn.SpatialCrossMapLRN(size, alpha, beta, k)
   local input = torch.rand(nbfeatures,inputSize,inputSize)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

   -- test batch mode and type
   local output = module:forward(input):clone()
   local gradOutput = output:clone():uniform(0,1)
   local gradInput = module:backward(input, gradOutput):clone()
   local batchSize = 4
   local input2 = torch.rand(batchSize,nbfeatures,inputSize,inputSize):float()
   input2[2]:copy(input)

   module:float() -- type-cast
   local output2 = module:forward(input2)
   local gradOutput2 = output2:clone():uniform(0,1)
   gradOutput2[2]:copy(gradOutput)
   local gradInput2 = module:backward(input2, gradOutput2)

   mytester:assertTensorEq(output2[2], output:float(), 0.000001, "SpatialCrossMapLRN 2d forward batch err")
   mytester:assertTensorEq(gradOutput2[2], gradOutput:float(), 0.000001, "SpatialCrossMapLRN 2d backward batch err")

   module:double()
   input2 = input2:double()
   local err = jac.testJacobian(module,input2)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input2)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


function nntest.SpatialConvolution()
   local from = math.random(1,5)
   local to = math.random(1,5)
   local ki = math.random(1,5)
   local kj = math.random(1,5)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(5,7)
   local outj = math.random(5,7)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local module = nn.SpatialConvolution(from, to, ki, kj, si, sj)
   local input = torch.Tensor(from, inj, ini):zero()

   -- stochastic

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err , precision, 'error on weight [direct update] ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err , precision, 'error on bias [direct update] ')

   nn.hessian.enable()

   local err = jac.testDiagHessianInput(module, input)
   mytester:assertlt(err , precision, 'error on diagHessianInput')

   local err = jac.testDiagHessianWeight(module, input)
   mytester:assertlt(err , precision, 'error on diagHessianWeight')

   local err = jac.testDiagHessianBias(module, input)
   mytester:assertlt(err , precision, 'error on diag HessianBias')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'error on bias [%s]', t))
   end

   -- batch

   --verbose = true
   local batch = math.random(2,5)
   outi = math.random(4,8)
   outj = math.random(4,8)
   ini = (outi-1)*si+ki
   inj = (outj-1)*sj+kj
   module = nn.SpatialConvolution(from, to, ki, kj, si, sj)
   input = torch.Tensor(batch,from,inj,ini):zero()

--    print(from, to, ki, kj, si, sj, batch, ini, inj)
--    print(module.weight:size())
--    print(module.gradWeight:size())

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'batch error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'batch error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'batch error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err , precision, 'batch error on weight [direct update] ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err , precision, 'batch error on bias [direct update] ')

   local err = jac.testDiagHessianInput(module, input)
   mytester:assertlt(err , precision, 'error on diagHessianInput')

   local err = jac.testDiagHessianWeight(module, input)
   mytester:assertlt(err , precision, 'error on diagHessianWeight')

   local err = jac.testDiagHessianBias(module, input)
   mytester:assertlt(err , precision, 'error on diag HessianBias')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'batch error on bias [%s]', t))
   end

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SpatialConvolutionMM()
   local from = math.random(2,5)
   local to = math.random(1,5)
   local ki = math.random(1,5)
   local kj = math.random(1,5)
   local di =  math.random(1,4)
   local dj =  math.random(1,4)
   local padW = math.random(0,2)
   local padH = math.random(0,2)
   local outi = math.random(5,9)
   local outj = math.random(5,9)
   local ini = (outi-1)*di+ki-padW*2
   local inj = (outj-1)*dj+kj-padH*2
   local module = nn.SpatialConvolutionMM(from, to, ki, kj, di, dj, padW, padH)
   local input = torch.Tensor(from, inj, ini):zero()

   -- stochastic

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err , precision, 'error on weight [direct update] ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err , precision, 'error on bias [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'error on bias [%s]', t))
   end

   -- batch

   --verbose = true
   local batch = math.random(2,5)

   module = nn.SpatialConvolutionMM(from, to, ki, kj, di, dj, padW, padH)
   input = torch.Tensor(batch,from,inj,ini):zero()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'batch error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'batch error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'batch error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err , precision, 'batch error on weight [direct update] ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err , precision, 'batch error on bias [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'batch error on bias [%s]', t))
   end

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')

   -- non-contiguous
   local input = torch.randn(batch,from,ini,inj):transpose(3,4) -- non-contiguous
   local inputc = input:contiguous() -- contiguous
   local output = module:forward(input):clone()
   local outputc = module:forward(inputc):clone()
   mytester:asserteq(0, (output-outputc):abs():max(), torch.typename(module) .. ' - contiguous err ')
   local gradInput = module:backward(input, output):clone()
   local gradInputc = module:backward(inputc, outputc):clone()
   mytester:asserteq(0, (gradInput-gradInputc):abs():max(), torch.typename(module) .. ' - contiguous err ')
end

function nntest.SpatialConvolutionLocal()
   local from = math.random(1,5)
   local to = math.random(1,5)
   local ki = math.random(1,5)
   local kj = math.random(1,5)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(5,7)
   local outj = math.random(5,7)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local module = nn.SpatialConvolutionLocal(from, to, ini, inj, ki, kj, si, sj)
   local input = torch.Tensor(from, inj, ini):zero()

   -- stochastic

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err , precision, 'error on weight [direct update] ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err , precision, 'error on bias [direct update] ')

   nn.hessian.enable()

   local err = jac.testDiagHessianInput(module, input)
   mytester:assertlt(err , precision, 'error on diagHessianInput')

   local err = jac.testDiagHessianWeight(module, input)
   mytester:assertlt(err , precision, 'error on diagHessianWeight')

   local err = jac.testDiagHessianBias(module, input)
   mytester:assertlt(err , precision, 'error on diag HessianBias')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'error on bias [%s]', t))
   end

   -- batch

   --verbose = true
   local batch = math.random(2,5)
   outi = math.random(4,8)
   outj = math.random(4,8)
   ini = (outi-1)*si+ki
   inj = (outj-1)*sj+kj
   module = nn.SpatialConvolutionLocal(from, to, ini, inj, ki, kj, si, sj)
   input = torch.Tensor(batch,from,inj,ini):zero()

--    print(from, to, ki, kj, si, sj, batch, ini, inj)
--    print(module.weight:size())
--    print(module.gradWeight:size())

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'batch error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'batch error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'batch error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err , precision, 'batch error on weight [direct update] ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err , precision, 'batch error on bias [direct update] ')

   local err = jac.testDiagHessianInput(module, input)
   mytester:assertlt(err , precision, 'error on diagHessianInput')

   local err = jac.testDiagHessianWeight(module, input)
   mytester:assertlt(err , precision, 'error on diagHessianWeight')

   local err = jac.testDiagHessianBias(module, input)
   mytester:assertlt(err , precision, 'error on diag HessianBias')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'batch error on bias [%s]', t))
   end

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')

   -- check against nn.SpatialConvolution
   local conv = nn.SpatialConvolution(from, to, ki, kj, si, sj)
   torch.repeatTensor(module.bias, conv.bias:view(to, 1, 1), 1, outi, outj)
   torch.repeatTensor(module.weight, conv.weight:view(1, 1, from, to, ki, kj), outi, outj, 1, 1, 1, 1)
   local input = torch.rand(batch, from, inj, ini)
   local output = module:forward(input)
   local outputConv = conv:forward(input)
   local err = torch.dist(output, outputConv)
   mytester:assertlt(err, precision, 'error checking against nn.SpatialConvolution')

end

function nntest.SpatialFullConvolution()
   local from = math.random(2,5)
   local to = math.random(1,5)
   local ki = math.random(1,5)
   local kj = math.random(1,5)
   local di =  math.random(1,4)
   local dj =  math.random(1,4)
   local padW = math.random(0,2)
   local padH = math.random(0,2)
   local outi = math.random(5,9)
   local outj = math.random(5,9)
   local adjW = (outi + padW*2 - ki) % di
   local adjH = (outj + padH*2 - kj) % dj
   local ini = math.floor((outi + padW*2 - ki)/di + 1)
   local inj = math.floor((outj + padH*2 - kj)/dj + 1)
   local module = nn.SpatialFullConvolution(from, to, ki, kj, di, dj, padW, padH, adjW, adjH)
   local input = torch.Tensor(from, inj, ini):zero()

   -- stochastic

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err , precision, 'error on weight [direct update] ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err , precision, 'error on bias [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'error on bias [%s]', t))
   end

   -- batch

   --verbose = true
   local batch = math.random(2,5)

   module = nn.SpatialFullConvolution(from, to, ki, kj, di, dj, padW, padH, adjW, adjH)
   input = torch.Tensor(batch,from,inj,ini):zero()

   -- Check that the required output size matches the actual output size
   local output = module:forward(input)
   mytester:asserteq(output:size(3), outj, 'output height error')
   mytester:asserteq(output:size(4), outi, 'output width error')

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'batch error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'batch error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'batch error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err , precision, 'batch error on weight [direct update] ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err , precision, 'batch error on bias [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'batch error on bias [%s]', t))
   end

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')

   -- non-contiguous
   local input = torch.randn(batch,from,ini,inj):transpose(3,4) -- non-contiguous
   local inputc = input:contiguous() -- contiguous
   local output = module:forward(input)
   local outputc = module:forward(inputc)
   mytester:asserteq(0, (output-outputc):abs():max(), torch.typename(module) .. ' - contiguous err ')
   local gradInput = module:backward(input, output)
   local gradInputc = module:backward(inputc, outputc)
   mytester:asserteq(0, (gradInput-gradInputc):abs():max(), torch.typename(module) .. ' - contiguous err ')
end

function nntest.SpatialConvolutionMap()
   local from = math.random(1,5)
   local fanin = math.random(1, from)
   local to = math.random(1,5)
   local ki = math.random(1,5)
   local kj = math.random(1,5)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local outi = math.random(5,9)
   local outj = math.random(5,9)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local module = nn.SpatialConvolutionMap(nn.tables.random(from, to, fanin), ki, kj, si, sj)
   local input = torch.Tensor(from, inj, ini):zero()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'error on bias ')

   nn.hessian.enable()

   local err = jac.testDiagHessianInput(module, input)
   mytester:assertlt(err , precision, 'error on diagHessianInput')

   local err = jac.testDiagHessianWeight(module, input)
   mytester:assertlt(err , precision, 'error on diagHessianWeight')

   local err = jac.testDiagHessianBias(module, input)
   mytester:assertlt(err , precision, 'error on diag HessianBias')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'error on bias [%s]', t))
   end

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')



    -- batch

   --verbose = true
   local batch = math.random(2,6)
   module = nn.SpatialConvolutionMap(nn.tables.random(from, to, fanin), ki, kj, si, sj)
   input = torch.Tensor(batch,from,inj,ini):zero()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'batch error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'batch error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'batch error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err , precision, 'batch error on weight [direct update] ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err , precision, 'batch error on bias [direct update] ')

   local err = jac.testDiagHessianInput(module, input)
   mytester:assertlt(err , precision, 'error on diagHessianInput')

   local err = jac.testDiagHessianWeight(module, input)
   mytester:assertlt(err , precision, 'error on diagHessianWeight')

   local err = jac.testDiagHessianBias(module, input)
   mytester:assertlt(err , precision, 'error on diag HessianBias')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'batch error on bias [%s]', t))
   end

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SpatialFullConvolutionMap()
   local from = math.random(2,4)
   local to = math.random(2,5)
   local fanin = math.random(1, from)
   local tt = nn.tables.random(from, to, fanin)
   local ki = math.random(2,5)
   local kj = math.random(2,5)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local ini = math.random(5,7)
   local inj = math.random(5,7)
   local module = nn.SpatialFullConvolutionMap(tt, ki, kj, si, sj)
   local input = torch.Tensor(from, inj, ini):zero()

   -- stochastic
      local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err , precision, 'error on weight [direct update] ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err , precision, 'error on bias [direct update] ')

   nn.hessian.enable()

   local err = jac.testDiagHessianInput(module, input)
   mytester:assertlt(err , precision, 'error on diagHessianInput')

   local err = jac.testDiagHessianWeight(module, input)
   mytester:assertlt(err , precision, 'error on diagHessianWeight')

   local err = jac.testDiagHessianBias(module, input)
   mytester:assertlt(err , precision, 'error on diag HessianBias')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'error on bias [%s]', t))
   end

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SpatialFullConvolutionCompare()
    local from = math.random(2,4)
    local to = math.random(2,5)
    local tt = nn.tables.full(from, to)
    local ki = math.random(2,5)
    local kj = math.random(2,5)
    local si = math.random(1,3)
    local sj = math.random(1,3)
    local ini = math.random(7,8)
    local inj = math.random(7,8)
    local module1 = nn.SpatialFullConvolutionMap(tt, ki, kj, si, sj)
    local module2 = nn.SpatialFullConvolution(from, to, ki, kj, si, sj)
    local input = torch.rand(from, inj, ini)
    for k=1,tt:size(1) do
       module1.weight[k]:copy(module2.weight[tt[k][1]][tt[k][2]])
       module1.bias:copy(module2.bias)
    end

    local o1 = module1:updateOutput(input)
    local o2 = module2:updateOutput(input)
    mytester:assertlt(o1:dist(o2), precision, 'error on output')

    local go1 = torch.rand(o1:size())
    local go2 = go1:clone()

    local gi1= module1:updateGradInput(input,go1)
    local gi2 = module2:updateGradInput(input,go2)
    mytester:assertlt(gi1:dist(gi2), precision, 'error on gradInput')

    module1:zeroGradParameters()
    module2:zeroGradParameters()

    module1:accGradParameters(input,go1)
    module2:accGradParameters(input,go2)
    for k=1,tt:size(1) do
      mytester:assertlt(module1.gradWeight[k]:dist(module2.gradWeight[tt[k][1]][tt[k][2]]),precision,'error on gradWeight ' .. k)
    end
    mytester:assertlt(module1.gradBias:dist(module2.gradBias),precision,'error on gradBias ')
end

local function batchcompare(smod, sin, plist)
   local bs = torch.LongStorage(sin:dim()+1)
   bs[1] = 1
   for i=1,sin:dim() do bs[i+1] = sin:size()[i] end
   local bin = torch.Tensor(bs):copy(sin)
   local bmod = smod:clone()

   local sout = smod:forward(sin):clone()
   local bout = bmod:forward(bin):clone()

   local sgout = torch.randn(sout:size())
   local bgout = torch.Tensor(bout:size())
   bgout:copy(sgout)

   local sgin = smod:backward(sin, sgout)
   local bgin = bmod:backward(bin, bgout)

   smod:accGradParameters(sin, sgout, 1)
   bmod:accGradParameters(bin, bgout, 1)

   mytester:assertTensorEq(sout,bout:select(1,1), 1e-8, 'batchcompare error on output')
   mytester:assertTensorEq(sgin,bgin:select(1,1), 1e-8, 'batchcompare error on gradInput')

   for i,v in pairs(plist) do
      mytester:assertTensorEq(smod[v],bmod[v], 1e-8, 'batchcompare error on ' .. v)
   end
end

function nntest.SpatialConvolutionBatchCompare()
   local from = math.random(1,5)
   local to = math.random(1,5)
   local ki = math.random(1,5)
   local kj = math.random(1,5)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(5,9)
   local outj = math.random(5,9)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local module = nn.SpatialConvolution(from, to, ki, kj, si, sj)
   module:zeroGradParameters()
   local input = torch.randn(from,inj,ini)

   batchcompare(module,input, {'weight','bias','gradWeight','gradBias'})
end

function nntest.SpatialFullConvolutionBatchCompare()
   local from = math.random(1,5)
   local to = math.random(1,5)
   local ki = math.random(1,5)
   local kj = math.random(1,5)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local ini = math.random(5,9)
   local inj = math.random(5,9)

   local module = nn.SpatialFullConvolution(from, to, ki, kj, si, sj)
   module:zeroGradParameters()
   local input = torch.randn(from, inj, ini)

   batchcompare(module,input, {'weight','bias','gradWeight','gradBias'})
end



function nntest.SpatialSubSamplingBatchCompare()
   local from = math.random(1,6)
   local ki = math.random(1,5)
   local kj = math.random(1,5)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(6,10)
   local outj = math.random(6,10)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local module = nn.SpatialSubSampling(from, ki, kj, si, sj)
   module:zeroGradParameters()
   local input = torch.randn(from,inj,ini)--torch.Tensor(from, inj, ini):zero()

   batchcompare(module,input, {'weight','bias','gradWeight','gradBias'})
end

function nntest.SpatialSubSampling()
   local from = math.random(1,6)
   local ki = math.random(1,5)
   local kj = math.random(1,5)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(6,10)
   local outj = math.random(6,10)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local module = nn.SpatialSubSampling(from, ki, kj, si, sj)
   local input = torch.Tensor(from, inj, ini):zero()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err , precision, 'error on weight [direct update] ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err , precision, 'error on bias [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'error on bias [%s]', t))
   end

   local batch = math.random(2,5)
   outi = math.random(4,8)
   outj = math.random(4,8)
   ini = (outi-1)*si+ki
   inj = (outj-1)*sj+kj
   module = nn.SpatialSubSampling(from, ki, kj, si, sj)
   input = torch.Tensor(batch,from,inj,ini):zero()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'batch error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'batch error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'batch error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err , precision, 'batch error on weight [direct update] ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err , precision, 'batch error on bias [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'batch error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'batch error on bias [%s]', t))
   end

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SpatialMaxPooling()
   for _,ceil_mode in pairs({true,false}) do
      local from = math.random(1,5)
      local ki = math.random(1,4)
      local kj = math.random(1,4)
      local si = math.random(1,3)
      local sj = math.random(1,3)
      local outi = math.random(4,5)
      local outj = math.random(4,5)
      local padW = math.min(math.random(0,1),math.floor(ki/2))
      local padH =  math.min(math.random(0,1),math.floor(kj/2))
      local ini = (outi-1)*si+ki-2*padW
      local inj = (outj-1)*sj+kj-2*padH

      local ceil_string = ceil_mode and 'ceil' or 'floor'
      local module = nn.SpatialMaxPooling(ki,kj,si,sj,padW,padH)
      if ceil_mode then module:ceil() else module:floor() end
      local input = torch.rand(from,inj,ini)

      local err = jac.testJacobian(module, input)
      mytester:assertlt(err, precision, 'error '..ceil_string..' mode on state ')

      local ferr, berr = jac.testIO(module, input)
      mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
      mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

      -- batch
      local nbatch = math.random(2,5)
      input = torch.rand(nbatch,from,inj,ini)
      module = nn.SpatialMaxPooling(ki,kj,si,sj,padW,padH)
      if ceil_mode then module:ceil() else module:floor() end

      local err = jac.testJacobian(module, input)
      mytester:assertlt(err, precision, 'error '..ceil_string..' mode on state (Batch)')

      local ferr, berr = jac.testIO(module, input)
      mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err (Batch) ')
      mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err (Batch) ')
  end
end

function nntest.SpatialMaxUnpooling()
   for _,ceil_mode in pairs({true,false}) do
      local from = math.random(1,5)
      local ki = math.random(2,4)
      local kj = math.random(2,4)
      local si, sj = ki, kj
      local outi = math.random(4,5)
      local outj = math.random(4,5)
      local padW = math.min(math.random(0,1),math.floor(ki/2))
      local padH = math.min(math.random(0,1),math.floor(kj/2))
      local ini = (outi-1)*si+ki-2*padW
      local inj = (outj-1)*sj+kj-2*padH

      local ceil_string = ceil_mode and 'ceil' or 'floor'
      local poolingModule = nn.SpatialMaxPooling(ki,kj,si,sj,padW,padH)
      if ceil_mode then poolingModule:ceil() else poolingModule:floor() end
      local module = nn.SpatialMaxUnpooling(poolingModule)

      local original = torch.rand(from,inj,ini)
      local input = poolingModule:forward(original)
      local output = module:forward(input)

      mytester:assert(output:isSameSizeAs(original),'SpatialMaxUnpooling output size err')

      local err = jac.testJacobian(module, input)
      mytester:assertlt(err, precision, 'error '..ceil_string..' mode on state ')

      local ferr, berr = jac.testIO(module, input)
      mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
      mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

      -- batch
      local nbatch = math.random(2,5)
      original = torch.rand(nbatch,from,inj,ini)
      input = poolingModule:forward(original)
      output = module:forward(input)

      mytester:assert(output:isSameSizeAs(original),'SpatialMaxUnpooling batch output size err')

      local err = jac.testJacobian(module, input)
      mytester:assertlt(err, precision, 'error '..ceil_string..' mode on state (Batch)')

      local ferr, berr = jac.testIO(module, input)
      mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err (Batch) ')
      mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err (Batch) ')
  end
end

function nntest.SpatialFractionalMaxPooling()
    local batch = math.random(1, 3)
    local plane = math.random(1, 3)
    local outW = math.random(1, 7)
    local outH = math.random(1, 7)
    local poolSizeW = math.random(2, 4)
    local poolSizeH = math.random(2, 4)

    local minInW = outW + poolSizeW
    local minInH = outH + poolSizeH

    local inW = math.random(minInW, minInW + 6)
    local inH = math.random(minInH, minInH + 6)

    -- fix the pooling regions so they aren't regenerated with every
    -- forward(), so testJacobian can work properly
    local module =
        nn.SpatialFractionalMaxPooling(poolSizeW, poolSizeH, outW, outH)
        :fixPoolingRegions()
    local input = nil
    if batch == 1 then
        input = torch.Tensor(plane, inH, inW):zero()
    else
        input = torch.Tensor(batch, plane, inH, inW):zero()
    end

    local err = nn.Jacobian.testJacobian(module, input)
    mytester:assertlt(err, precision, 'error on state')
end

function nntest.SpatialFractionalMaxPooling_Ratio()
    -- Fix a reduction ratio, and test with two different input sizes
    local reductionRatioW = torch.uniform(0.4, 0.74)
    local reductionRatioH = torch.uniform(0.4, 0.74)

    for tries = 1, 2 do
        local batch = math.random(1, 3)
        local plane = math.random(1, 3)
        local poolSizeW = math.random(2, 3)
        local poolSizeH = math.random(2, 3)

        local minInW = math.random(5, 8) + poolSizeW
        local minInH = math.random(5, 8) + poolSizeH

        local inW = math.random(minInW, minInW + 6)
        local inH = math.random(minInH, minInH + 6)

        -- fix the pooling regions so they aren't regenerated with every
        -- forward(), so testJacobian can work properly
        local module =
            nn.SpatialFractionalMaxPooling(poolSizeW, poolSizeH,
                                           reductionRatioW, reductionRatioH)
            :fixPoolingRegions()
        local input = nil
        if batch == 1 then
            input = torch.Tensor(plane, inH, inW):zero()
        else
            input = torch.Tensor(batch, plane, inH, inW):zero()
        end

        -- Make sure that the output size is based on our ratio
        local output = module:updateOutput(input)
        if batch == 1 then
            mytester:asserteq(output:size(3), math.floor(reductionRatioW * inW))
            mytester:asserteq(output:size(2), math.floor(reductionRatioH * inH))
        else
            mytester:asserteq(output:size(4), math.floor(reductionRatioW * inW))
            mytester:asserteq(output:size(3), math.floor(reductionRatioH * inH))
        end

        local err = nn.Jacobian.testJacobian(module, input)
        mytester:assertlt(err, precision, 'error on state')
    end
end

function nntest.SpatialAveragePooling()
   for _,count_include_pad in pairs({true,false}) do
      for _,ceil_mode in pairs({true,false}) do
        local from = math.random(1,5)
        local ki = math.random(1,4)
        local kj = math.random(1,4)
        local si = math.random(1,3)
        local sj = math.random(1,3)
        local outi = math.random(4,5)
        local outj = math.random(4,5)
        local padW = math.min(math.random(0,1),math.floor(ki/2))
        local padH =  math.min(math.random(0,1),math.floor(kj/2))
        local ini = (outi-1)*si+ki-2*padW
        local inj = (outj-1)*sj+kj-2*padH

        local mode_string = ceil_mode and 'ceil' or 'floor'

        local module = nn.SpatialAveragePooling(ki, kj, si, sj, padW, padH)
        if ceil_mode then module:ceil() else module:floor() end
        if count_include_pad then
           module:setCountIncludePad()
           mode_string = mode_string .. ' - count include padding'
        else
           module:setCountExcludePad()
           mode_string = mode_string .. ' - count exclude padding'
        end
        local input = torch.Tensor(from, inj, ini):uniform()

        local err = jac.testJacobian(module, input)
        mytester:assertlt(err, precision, 'error'..mode_string..' on state ')

        local ferr, berr = jac.testIO(module, input)
        mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
        mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

        -- batch
        local batch = math.random(2,5)
        outi = math.random(4,5)
        outj = math.random(4,5)
        local padW = math.min(math.random(0,1),math.floor(ki/2))
        local padH =  math.min(math.random(0,1),math.floor(kj/2))
        local ini = (outi-1)*si+ki-2*padW
        local inj = (outj-1)*sj+kj-2*padH

        module = nn.SpatialAveragePooling(ki, kj, si, sj, padW, padH)
        if ceil_mode then module:ceil() else module:floor() end
        if count_include_pad then
           module:setCountIncludePad()
        else
           module:setCountExcludePad()
        end
        input = torch.Tensor(batch,from,inj,ini):uniform()

        local err = jac.testJacobian(module, input)
        mytester:assertlt(err, precision, 'batch error'..mode_string..' on state ')

        local ferr, berr = jac.testIO(module, input)
        mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
        mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')

        local ferr, berr = jac.testIO(module, input)
        mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err (Batch) ')
        mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err (Batch) ')

      end
   end
   -- test against SpatialSubSampling
   local from = math.random(1,6)
   local ki = math.random(1,5)
   local kj = math.random(1,5)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(6,10)
   local outj = math.random(6,10)
   local padW = 0
   local padH = 0
   local ini = (outi-1)*si+ki-2*padW
   local inj = (outj-1)*sj+kj-2*padH

   local module = nn.SpatialAveragePooling(ki, kj, si, sj, padW, padH)
   local sap = nn.SpatialSubSampling(from, ki, kj, si, sj)
   sap.weight:fill(1.0/(ki*kj))
   sap.bias:fill(0.0)

   local input = torch.Tensor(from, inj, ini):uniform()

   local output = module:forward(input)
   local gradInput = module:backward(input, output)
   local output2 = sap:forward(input)
   local gradInput2 = sap:updateGradInput(input, output)

   mytester:assertTensorEq(output, output2, 0.000001, torch.typename(module) .. ' forward err ')
   mytester:assertTensorEq(gradInput, gradInput2, 0.000001, torch.typename(module) .. ' backward err ')

   -- test against SpatialSubSampling, batch mode
   local batch = math.random(2,5)
   outi = math.random(4,8)
   outj = math.random(4,8)
   local padW = 0
   local padH = 0
   local ini = (outi-1)*si+ki-2*padW
   local inj = (outj-1)*sj+kj-2*padH

   module = nn.SpatialAveragePooling(ki, kj, si, sj, padW, padH)
   input = torch.Tensor(batch,from,inj,ini):uniform()

   local sap = nn.SpatialSubSampling(from, ki, kj, si, sj)
   sap.weight:fill(1.0/(ki*kj))
   sap.bias:fill(0.0)

   local output = module:forward(input)
   local gradInput = module:backward(input, output)
   local output2 = sap:forward(input)
   local gradInput2 = sap:updateGradInput(input, output)

   mytester:assertTensorEq(output, output2, 0.000001, torch.typename(module) .. ' forward err (Batch) ')
   mytester:assertTensorEq(gradInput, gradInput2, 0.000001, torch.typename(module) .. ' backward err (Batch) ')

end

function nntest.SpatialAdaptiveMaxPooling()
   local from = math.random(1,5)
   local ki = math.random(1,5)
   local kj = math.random(1,5)
   local ini = math.random(1,16)
   local inj = math.random(1,16)

   local module = nn.SpatialAdaptiveMaxPooling(ki,kj)
   local input = torch.rand(from,ini,inj)

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

   -- batch
   local nbatch = math.random(1,3)
   input = torch.rand(nbatch,from,ini,inj)
   module = nn.SpatialAdaptiveMaxPooling(ki,kj)

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state (Batch) ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err (Batch) ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err (Batch) ')

   -- non-contiguous

   input = torch.rand(from,ini,inj):transpose(2,3)
   module = nn.SpatialAdaptiveMaxPooling(ki,kj)
   local inputc = input:contiguous() -- contiguous
   local output = module:forward(input):clone()
   local outputc = module:forward(inputc):clone()
   mytester:asserteq(0, (output-outputc):abs():max(), torch.typename(module) .. ' - non-contiguous err ')
   local gradInput = module:backward(input, output):clone()
   local gradInputc = module:backward(inputc, outputc):clone()
   mytester:asserteq(0, (gradInput-gradInputc):abs():max(), torch.typename(module) .. ' - non-contiguous err ')

   -- non-contiguous batch
   local nbatch = math.random(1,3)
   input = torch.rand(nbatch,from,ini,inj):transpose(1,3):transpose(2,4)
   local inputc = input:contiguous() -- contiguous
   module = nn.SpatialAdaptiveMaxPooling(ki,kj)

   local output = module:forward(input):clone()
   local outputc = module:forward(inputc):clone()
   mytester:asserteq(0, (output-outputc):abs():max(), torch.typename(module) .. ' - batch non-contiguous err ')
   local gradInput = module:backward(input, output):clone()
   local gradInputc = module:backward(inputc, outputc):clone()
   mytester:asserteq(0, (gradInput-gradInputc):abs():max(), torch.typename(module) .. ' - batch non-contiguous err ')

end

function nntest.SpatialLPPooling()
   local fanin = math.random(1,4)
   local osizex = math.random(1,4)
   local osizey = math.random(1,4)
   local p = 2
   local mx = math.random(2,6)
   local my = math.random(2,6)
   local dx = math.random(2,mx)
   local dy = math.random(2,my)
   local sizex = osizex*mx
   local sizey = osizey*my
   local module = nn.SpatialLPPooling(fanin,p,mx,my,dx,dy)
   local input = torch.rand(fanin,sizey,sizex)

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Sum()
   -- 1D
   local ini = math.random(3,7)
   local input = torch.Tensor(ini):zero()
   local module = nn.Sum(1)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   -- negative dimension
   local module   = nn.Sum(-1)
   local input    = torch.Tensor({1, 2, 3})
   local expected = torch.Tensor({6})
   local output   = module:forward(input)
   mytester:assertlt(torch.norm(output-expected), precision, 'error on forward ')

   -- batch
   local dimension = 1
   local module    = nn.Sum(dimension, 1)
   local input     = torch.Tensor({{1, 2, 3},{4, 5, 6}})
   local expected  = torch.Tensor({6, 15})
   local output    = module:forward(input)
   mytester:assertlt(torch.norm(output-expected), precision, 'error on forward ')

   local err       = jac.testJacobian(module, input)
   mytester:assertlt(err,precision, 'error on state ')

   -- mean + batch
   local dimension = 1
   local module    = nn.Sum(dimension, 1, true)
   local input     = torch.Tensor({{1, 2, 3},{4, 5, 6}})
   local expected  = input:mean(dimension + 1)
   local output    = module:forward(input)

   mytester:assertlt(torch.norm(output-expected), precision, 'error on forward ')

   local err       = jac.testJacobian(module, input)
   mytester:assertlt(err,precision, 'error on state ')

   -- 3D
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.Sum(torch.random(1,3))

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Tanh()
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.Tanh()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision ,  'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.TemporalConvolution()
   -- 1D
   local from = math.random(1,5)
   local to = math.random(1,5)
   local ki = math.random(1,5)
   local si = math.random(1,4)
   local outi = math.random(5,7)
   local ini = (outi-1)*si+ki
   local module = nn.TemporalConvolution(from, to, ki,si)
   local input = torch.Tensor(ini, from):zero()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err , precision, 'error on weight [direct update]')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err , precision, 'error on bias [direct update]')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'error on bias [%s]', t))
   end

   -- 2D
   local nBatchFrame = 4
   local input = torch.Tensor(nBatchFrame, ini, from):zero()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err , precision, 'error on weight [direct update]')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err , precision, 'error on bias [direct update]')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'error on bias [%s]', t))
   end

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')

   -- 2D matches 1D
   local output = module:forward(input):clone()
   local outputGrad = torch.randn(output:size())
   local inputGrad = module:backward(input, outputGrad):clone()

   local input1D = input:select(1, 2)
   local output1D = module:forward(input1D)
   local outputGrad1D = outputGrad:select(1, 2)
   local inputGrad1D = module:backward(input1D, outputGrad1D)

   mytester:assertTensorEq(output:select(1,2), output1D, 0.000001, 'error on 2D vs 1D forward)')
   mytester:assertTensorEq(inputGrad:select(1,2), inputGrad1D, 0.000001, 'error on 2D vs 1D backward)')
end

function nntest.TemporalSubSampling()
   local from = math.random(1,5)
   local ki = math.random(1,6)
   local si = math.random(1,4)
   local outi = math.random(6,9)
   local ini = (outi-1)*si+ki
   local module = nn.TemporalSubSampling(from, ki, si)
   local input = torch.Tensor(ini, from):zero()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err , precision, 'error on weight [direct update] ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err , precision, 'error on bias [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'error on bias [%s]', t))
   end

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.TemporalMaxPooling()
   local from = math.random(2,4)
   local ki = math.random(5,7)
   local si = math.random(1,2)
   local outi = math.random(30,40)
   local ini = (outi-1)*si+ki
   local module = nn.TemporalMaxPooling(ki, si)
   local input = torch.Tensor(ini, from):zero()

   -- 1D
   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')

   -- 2D
   local nBatchFrame = 2
   local input = torch.Tensor(nBatchFrame, ini, from):zero()
   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')

   -- 2D matches 1D
   local output = module:forward(input):clone()
   local outputGrad = torch.randn(output:size())
   local inputGrad = module:backward(input, outputGrad):clone()

   local input1D = input:select(1, 2)
   local output1D = module:forward(input1D)
   local outputGrad1D = outputGrad:select(1, 2)
   local inputGrad1D = module:backward(input1D, outputGrad1D)

   mytester:assertTensorEq(output:select(1,2), output1D, 0.000001, 'error on 2D vs 1D forward)')
   mytester:assertTensorEq(inputGrad:select(1,2), inputGrad1D, 0.000001, 'error on 2D vs 1D backward)')
end

function nntest.VolumetricFullConvolution_simple_test()
    local module = nn.VolumetricFullConvolution(3, 1, 3, 3, 3, 3, 3, 3);
    module.weight:fill(1);
    module.bias:fill(0.1);

    local input = torch.Tensor(1, 3, 2, 2, 2):zero();
    for c = 1,3 do
        input[1][c][1][1][1] = 1
    end
    local output = module:forward(input)
    for t = 1,6 do
        for h = 1,6 do
            for w = 1,6 do
                if t <= 3 and h <= 3 and w <= 3 then
                    mytester:assertlt(output[1][1][t][h][w] - 3.1, precision, 'error on forward ')
                else
                    mytester:assertlt(output[1][1][t][h][w] - 0.1, precision, 'error on forward ')
                end
            end
        end
    end

    local gradOut = torch.Tensor(1, 1, 6, 6, 6):fill(0.1);
    local gradIn = module:backward(input, gradOut)
    for t = 1,2 do
        for h = 1,2 do
            for w = 1,2 do
                mytester:assertlt(gradIn[1][1][t][h][w] - 2.7, precision,
                                  'error on backward input gradients ')
            end
        end
    end

    mytester:assertlt(module.gradBias[1] - 21.6, precision,
                      'error on backward gradBias ')
    for c = 1,3 do
        for t = 1,3 do
            for h = 1,3 do
                for w = 1,3 do
                    mytester:assertlt(module.gradWeight[1][c][t][h][w] - 0.1, precision,
                                      'error on backward weight gradients ')
                end
            end
        end
    end
end

function nntest.VolumetricFullConvolution()
    local from = math.random(2,3)
    local to = math.random(2,3)
    local kt = math.random(3,4)
    local ki = math.random(3,4)
    local kj = ki
    local st = math.random(1,3)
    local si = math.random(1,3)
    local sj = si
    local int = math.random(3,4)
    local ini = math.random(3,4)
    local inj = math.random(3,4)
    local bs = math.random(1, 6)
    local module = nn.VolumetricFullConvolution(from, to, kt, ki, kj, st, si, sj)

    local input = torch.Tensor(bs, from, int, ini, inj):zero()

    local err = jac.testJacobian(module, input)
    mytester:assertlt(err, precision, 'error on state ')

    local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
    mytester:assertlt(err , precision, 'error on weight ')

    local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
    mytester:assertlt(err , precision, 'error on bias ')

    local ferr, berr = jac.testIO(module, input)
    mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
    mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end



function nntest.VolumetricConvolution()
   local from = math.random(2,5)
   local to = math.random(1,5)
   local kt = math.random(1,5)
   local ki = math.random(1,5)
   local kj = math.random(1,5)
   local st = math.random(1,4)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local padT = math.random(0,2)
   local padW = math.random(0,2)
   local padH = math.random(0,2)
   local outt = math.random(5,9)
   local outi = math.random(5,9)
   local outj = math.random(5,9)
   local int = (outt-1)*st+kt-padT*2
   local ini = (outi-1)*si+ki-padW*2
   local inj = (outj-1)*sj+kj-padH*2
   local module = nn.VolumetricConvolution(from, to, kt, ki, kj, st, si, sj, padT, padW, padH)
   local input = torch.Tensor(from, int, inj, ini):zero()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err , precision, 'error on weight [direct update] ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err , precision, 'error on bias [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'error on bias [%s]', t))
   end

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.VolumetricConvolutionBatchCompare()
   local from = math.random(2,3)
   local to = math.random(2,3)
   local kt = math.random(3,4)
   local ki = math.random(3,4)
   local kj = math.random(3,4)
   local st = math.random(2,3)
   local si = math.random(2,3)
   local sj = math.random(2,3)
   local padT = math.random(0,2)
   local padW = math.random(0,2)
   local padH = math.random(0,2)
   local outt = math.random(3,4)
   local outi = math.random(3,4)
   local outj = math.random(3,4)
   local int = (outt-1)*st+kt-padT*2
   local ini = (outi-1)*si+ki-padW*2
   local inj = (outj-1)*sj+kj-padH*2
   local module = nn.VolumetricConvolution(from, to, kt, ki, kj, st, si, sj, padT, padW, padH)
   module:zeroGradParameters()
   local input = torch.randn(from, int, inj, ini)
   batchcompare(module,input, {'weight','bias','gradWeight','gradBias'})
end

function nntest.VolumetricAveragePooling()
   local from = math.random(2,3)
   local kt = math.random(3,4)
   local ki = math.random(3,4)
   local kj = math.random(3,4)
   local st = math.random(2,3)
   local si = math.random(2,3)
   local sj = math.random(2,3)
   local outt = math.random(3,4)
   local outi = math.random(3,4)
   local outj = math.random(3,4)
   local int = (outt-1)*st+kt
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local module = nn.VolumetricAveragePooling(kt, ki, kj, st, si, sj)
   local input = torch.Tensor(from, int, inj, ini):zero()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')

      -- batch
   local nbatch = math.random(2,3)
   module = nn.VolumetricAveragePooling(kt, ki, kj, st, si, sj)
   input = torch.Tensor(nbatch, from, int, inj, ini):zero()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state (Batch) ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err (Batch) ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err (Batch) ')
end

function nntest.VolumetricMaxPooling()
   local from = math.random(2,3)
   local kt = math.random(3,4)
   local ki = math.random(3,4)
   local kj = math.random(3,4)
   local st = math.random(2,3)
   local si = math.random(2,3)
   local sj = math.random(2,3)
   local outt = math.random(3,4)
   local outi = math.random(3,4)
   local outj = math.random(3,4)
   local padT = math.min(math.random(0,2),math.floor(kt/2))
   local padW = math.min(math.random(0,2),math.floor(ki/2))
   local padH =  math.min(math.random(0,2),math.floor(kj/2))
   local int = (outt-1)*st+kt-2*padT
   local ini = (outi-1)*si+ki-2*padW
   local inj = (outj-1)*sj+kj-2*padH
   local module = nn.VolumetricMaxPooling(kt, ki, kj, st, si, sj, padT, padW, padH)
   local input = torch.Tensor(from, int, inj, ini):zero()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')

   -- batch
   local nbatch = math.random(2,3)
   module = nn.VolumetricMaxPooling(kt, ki, kj, st, si, sj, padT, padW, padH)
   input = torch.Tensor(nbatch, from, int, inj, ini):zero()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state (Batch) ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err (Batch) ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err (Batch) ')
end

function nntest.VolumetricMaxUnpooling()
   local from = math.random(2,3)
   local kt = math.random(3,4)
   local ki = math.random(3,4)
   local kj = math.random(3,4)
   local st, si, sj = kt, ki, kj
   local outt = math.random(3,4)
   local outi = math.random(3,4)
   local outj = math.random(3,4)
   local padT = math.min(math.random(0,2),math.floor(kt/2))
   local padW = math.min(math.random(0,2),math.floor(ki/2))
   local padH = math.min(math.random(0,2),math.floor(kj/2))
   local int = (outt-1)*st+kt-2*padT
   local ini = (outi-1)*si+ki-2*padW
   local inj = (outj-1)*sj+kj-2*padH

   local poolingModule = nn.VolumetricMaxPooling(kt, ki, kj, st, si, sj, padT, padW, padH)
   local module = nn.VolumetricMaxUnpooling(poolingModule)

   local original = torch.rand(from,int,inj,ini)
   local input = poolingModule:forward(original)
   local output = module:forward(input)
   mytester:assert(output:isSameSizeAs(original),'VolumetricMaxUnpooling output size err')

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

   -- batch
   local nbatch = math.random(2,3)
   original = torch.rand(nbatch,from,int,inj,ini)
   input = poolingModule:forward(original)
   output = module:forward(input)

   mytester:assert(output:isSameSizeAs(original),'VolumetricMaxUnpooling batch output size err')

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on Batch')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err (Batch) ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err (Batch) ')
end

function nntest.VolumetricMaxPooling_boundary()
   -- simple kernel 2x2x2 with striding 2x2x2
   local module = nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2):ceil()
   local nip = math.random(3,256)
   local input = torch.rand(nip, 2, 7, 7)

   -- do a forward pass
   local output = module:forward(input)

   -- checking output size
   mytester:asserteq(output:size(1), nip, 'wrong output channels')
   mytester:asserteq(output:size(2), 1, 'wrong output temporal length')
   mytester:asserteq(output:size(3), 4, 'wrong output height')
   mytester:asserteq(output:size(4), 4, 'wrong output width')

   -- checking output signals at top right
   for c = 1,nip do
      local max_val = input[c][1][1][7]
      for t = 1,2 do
        for h = 1,2 do
          max_val = math.max(max_val, input[c][t][h][7])
        end
      end
      mytester:asserteq(output[c][1][1][4], max_val, 'wrong forward execution')
   end
   -- checking output signals at bottom left
   for c = 1,nip do
       local max_val = input[c][1][7][1]
       for t = 1,2 do
         for w = 1,2 do
           max_val = math.max(max_val, input[c][t][7][w])
         end
       end
       mytester:asserteq(output[c][1][4][1], max_val, 'wrong forward execution')
   end

   -- check output signals at right bottom
    for c = 1,nip do
      local max_val = math.max(input[c][1][7][7], input[c][2][7][7])
      mytester:asserteq(output[c][1][4][4], max_val, 'wrong forward execution')
    end


   -- backward is supposed to be tested in nntest.VolumetricMaxPooling
   -- This is only test the boundary cases
end

function nntest.Module_getParameters_1()
   local n = nn.Sequential()
   n:add( nn.Linear(10,10) )
   local p = n:getParameters()

   mytester:asserteq((p[{ {1,100} }] - n.modules[1].weight):norm(), 0, 'getParameters(): weights wrong')
   mytester:asserteq((p[{ {101,110} }] - n.modules[1].bias):norm(), 0, 'getParameters(): bias wrong')
end

function nntest.Module_getParameters_2()
   local n = nn.Sequential()
   n:add( nn.Linear(10,10) )
   local _ = n:getParameters()

   n:add( nn.Linear(10,10) )
   local p = n:getParameters()

   mytester:asserteq((p[{ {111,210} }] - n.modules[2].weight):norm(), 0, 'error when appending new module')
   mytester:asserteq((p[{ {211,220} }] - n.modules[2].bias):norm(), 0, 'error when appending new module')
end

function nntest.Module_getParameters_3()
   local n = nn.Sequential()
   n:add( nn.Linear(10,10) )
   n:add( n.modules[1]:clone() )
   local p = n:getParameters()

   mytester:asserteq((p[{ {1,100} }] - n.modules[1].weight):norm(), 0, 'error when using cloning')
   mytester:asserteq((p[{ {101,110} }] - n.modules[1].bias):norm(), 0, 'error when using cloning')

   mytester:asserteq((p[{ {111,210} }] - n.modules[2].weight):norm(), 0, 'error when using cloning')
   mytester:asserteq((p[{ {211,220} }] - n.modules[2].bias):norm(), 0, 'error when using cloning')

   mytester:asserteq((p[{ {111,210} }] - n.modules[1].weight):norm(), 0, 'error when using cloning')
   mytester:asserteq((p[{ {211,220} }] - n.modules[1].bias):norm(), 0, 'error when using cloning')

   n:reset()

   mytester:assertgt((p[{ {111,210} }] - n.modules[1].weight):norm(), 0, 'error when using cloning')
   mytester:assertgt((p[{ {211,220} }] - n.modules[1].bias):norm(), 0, 'error when using cloning')
end

function nntest.Module_getParameters_4()
   local n = nn.Sequential()
   n:add( nn.Linear(10,10) )
   n:add( n.modules[1]:clone() )
   local _ = n:getParameters()

   n:add(nn.Linear(10,10))
   local p = n:getParameters()

   mytester:asserteq((p[{ {1,100} }] - n.modules[1].weight):norm(), 0, 'error when using cloning')
   mytester:asserteq((p[{ {101,110} }] - n.modules[1].bias):norm(), 0, 'error when using cloning')

   mytester:asserteq((p[{ {111,210} }] - n.modules[2].weight):norm(), 0, 'error when using cloning')
   mytester:asserteq((p[{ {211,220} }] - n.modules[2].bias):norm(), 0, 'error when using cloning')

   mytester:asserteq((p[{ {221,320} }] - n.modules[3].weight):norm(), 0, 'error when using cloning')
   mytester:asserteq((p[{ {321,330} }] - n.modules[3].bias):norm(), 0, 'error when using cloning')

   mytester:asserteq(p:nElement(), 3*(10*10+10), 'error: incorrect number of elements in flat vector')
end

function nntest.Module_getParameters_5()
   local n = nn.Sequential()
   n:add( nn.Linear(10,10) )
   n:add( n.modules[1]:clone('weight','bias') )
   local p = n:getParameters()

   mytester:asserteq((p[{ {1,100} }] - n.modules[1].weight):norm(), 0, 'error when using cloning+sharing')
   mytester:asserteq((p[{ {101,110} }] - n.modules[1].bias):norm(), 0, 'error when using cloning+sharing')

   mytester:asserteq((p[{ {1,100} }] - n.modules[2].weight):norm(), 0, 'error when using cloning+sharing')
   mytester:asserteq((p[{ {101,110} }] - n.modules[2].bias):norm(), 0, 'error when using cloning+sharing')

   n:reset()

   mytester:asserteq((p[{ {1,100} }] - n.modules[2].weight):norm(), 0, 'error when using cloning+sharing')
   mytester:asserteq((p[{ {101,110} }] - n.modules[2].bias):norm(), 0, 'error when using cloning+sharing')

   mytester:asserteq(p:nElement(), (10*10+10), 'error: incorrect number of elements in flat vector')
end

function nntest.Module_getParameters_6()
   local n = nn.Sequential()
   n:add( nn.Linear(10,10) )
   n:add( n.modules[1]:clone('weight','bias') )
   local _ = n:getParameters()

   n:add(nn.Linear(10,10))
   local p = n:getParameters()

   mytester:asserteq((p[{ {1,100} }] - n.modules[1].weight):norm(), 0, 'error when using cloning+sharing')
   mytester:asserteq((p[{ {101,110} }] - n.modules[1].bias):norm(), 0, 'error when using cloning+sharing')

   mytester:asserteq((p[{ {1,100} }] - n.modules[2].weight):norm(), 0, 'error when using cloning+sharing')
   mytester:asserteq((p[{ {101,110} }] - n.modules[2].bias):norm(), 0, 'error when using cloning+sharing')

   mytester:asserteq((p[{ {111,210} }] - n.modules[3].weight):norm(), 0, 'error when using cloning+sharing')
   mytester:asserteq((p[{ {211,220} }] - n.modules[3].bias):norm(), 0, 'error when using cloning+sharing')

   mytester:asserteq(p:nElement(), 2*(10*10+10), 'error: incorrect number of elements in flat vector')
end

function nntest.Module_getParameters_7()
   local n = nn.Sequential()
   n:add( nn.Linear(10,10) )
   n:add( n.modules[1]:clone('weight','bias') )
   local _ = n:getParameters()

   n:add(nn.Linear(10,10))
   local _ = n:getParameters()

   local n1 = nn.Sequential()
   n1:add( nn.Linear(10,10) )

   local n2 = nn.Sequential()
   n2:add( nn.Linear(10,10) )

   local n = nn.Sequential()
   n:add( n1 )
   n:add( n2 )

   local _ = n:getParameters()

   local nf = nn.Sequential()
   nf:add( n1 )
   nf:add( nn.Linear(10,1) )

   local p = nf:getParameters()

   mytester:asserteq((p[{ {1,100} }] - n1.modules[1].weight):norm(), 0, 'error when using cloning+partial realloc')
   mytester:asserteq((p[{ {101,110} }] - n1.modules[1].bias):norm(), 0, 'error when using cloning+partial realloc')

   mytester:asserteq((p[{ {111,120} }] - nf.modules[2].weight):norm(), 0, 'error when using cloning+partial realloc')
   mytester:asserteq((p[{ {121,121} }] - nf.modules[2].bias):norm(), 0, 'error when using cloning+partial realloc')

   mytester:asserteq(p:nElement(), 121, 'error: incorrect number of elements in flat vector')
end

function nntest.Module_getParameters_8()
   local function makeMLP(nin, ns)
      local net = nn.Sequential()

      for k,v in ipairs(ns) do
         net:add(nn.Linear(nin, v))
         nin = v
      end
      local _,_ = net:getParameters()
      return net
   end

  local mlp1 = makeMLP(10, {10,10})
  local mlp2 = makeMLP(10, {10,10})

  local net = nn.Sequential():add(mlp1:get(1))
                             :add(mlp2:get(1))

  -- clone the second MLP to ensure that the weights before calling getParameters are preserved
  mlp2 = mlp2:clone()

  local p, _ = net:getParameters()

  mytester:asserteq((p[{ {1,100} }] - net.modules[1].weight):norm(), 0, 'error when using partial realloc')
  mytester:asserteq((p[{ {111,210} }] - net.modules[2].weight):norm(), 0, 'error when using partial realloc')
  -- check that the weights have the same values as before get Parameters was called
  mytester:asserteq((net.modules[1].weight - mlp1.modules[1].weight):norm(), 0, ' error when using partial realloc')
  mytester:asserteq((net.modules[2].weight - mlp2.modules[1].weight):norm(), 0, ' error when using partial realloc')

end

function nntest.Module_getParameters_10()
  -- tensors are non-contiguous but compact; they can be gathered
  local L = nn.Linear(10,10)
  L.weight = torch.Tensor(10,10):t():fill(1)
  local tmp = torch.Tensor(10,10):fill(2)
  L.bias = tmp:select(1,2)
  local P = L:getParameters()
  mytester:asserteq(L.weight:mean(), 1)
  mytester:asserteq(L.bias:mean(), 2)
  mytester:asserteq(L.weight:storage(), L.bias:storage())
  mytester:asserteq(P:nElement(), 110)
  mytester:asserteq(P:storage():size(), 110)
  mytester:assertlt(L.bias[{ {10} }]:storageOffset() - 1, L.bias:storage():size())
end

function nntest.Module_getParameters_11()
  -- tensors are non-compact; they can't be gathered
  local L = nn.Linear(10,10)
  local tmp = torch.Tensor(10,10):fill(2)
  L.bias = tmp:select(2,2)
  local ok, err = pcall(L.getParameters, L)
  mytester:assert(not ok)
end

function nntest.Module_getParameters_12()
  -- tensors are expanded (i.e. have dimension 0)
  local L = nn.Linear(10,10)
  L.weight = torch.Tensor(10, 1):fill(1)
  torch.expand(L.weight, 10, 10)
  L.bias = torch.Tensor(10):fill(2)
  local P = L:getParameters()
  mytester:asserteq(L.weight:mean(), 1)
  mytester:asserteq(L.bias:mean(), 2)
  mytester:asserteq(L.weight:storage(), L.bias:storage())
  mytester:asserteq(P:nElement(), 20)
  mytester:asserteq(P:storage():size(), 20)
  mytester:assertlt(L.bias[{ {10} }]:storageOffset() - 1, L.bias:storage():size())
end

function nntest.Module_listModules()
   local batchSize = 4
   local inputSize, outputSize = 7, 6
   local linear = nn.Linear(inputSize, outputSize)
   local tanh = nn.Tanh()
   local reshape = nn.Reshape(outputSize/2, 2)
   local mlp3 = nn.Sequential()
   mlp3:add(linear)
   mlp3:add(tanh)
   mlp3:add(reshape)

   local mlp2 = nn.Sequential()
   local view = nn.View(outputSize)
   local linear2 = nn.Linear(outputSize, inputSize)
   local tanh2 = nn.Tanh()
   mlp2:add(mlp3)
   mlp2:add(view)
   mlp2:add(linear2)
   mlp2:add(tanh2)

   local concat = nn.ConcatTable()
   local id = nn.Identity()
   concat:add(mlp2)
   concat:add(id)
   local mlp = nn.Sequential()
   local add = nn.CAddTable()
   mlp:add(concat)
   mlp:add(add)

   local modules2 = {mlp, concat, mlp2, mlp3, linear, tanh, reshape, view, linear2, tanh2, id, add}
   local modules = mlp:listModules()

   mytester:assert(#modules2 == #modules, 'missing modules error')

   for i,module in ipairs(modules) do
      mytester:assert(torch.type(module) == torch.type(modules2[i]), 'module error')
   end
end

function nntest.PairwiseDistance()
   -- Note: testJacobian doesn't support table inputs, and rather than re-write
   -- it so that it does, I'll just use a split table module on the input.
   -- I assume both SplitTable and Sequential do not have bugs, otherwise this
   -- test will break.
   for p = 1,4 do  -- test a few Lp norms
      -- TEST CASE 1: non-batch input, same code path but includes a resize
      local ini = math.random(3,5)
      local input = torch.Tensor(2, ini):zero()
      local module = nn.Sequential()
      module:add(nn.SplitTable(1))
      module:add(nn.PairwiseDistance(p))

      local err = jac.testJacobian(module,input)
      mytester:assertlt(err, 1e-4, ' error on state ')

      local ferr,berr = jac.testIO(module,input)
      mytester:asserteq(ferr, 0, torch.typename(module)..' - i/o forward err ')
      mytester:asserteq(berr, 0, torch.typename(module)..' - i/o backward err ')

      -- Also check that the forward prop result is correct.
      input = torch.rand(2, ini)
      err = torch.dist(input:select(1,1), input:select(1,2), p) -
        module:forward(input)[1]
      mytester:assertlt(err,precision, ' error on non-batch fprop ')

      -- TEST CASE 2: batch input
      local inj = math.random(3,5)
      input = torch.Tensor(2, inj, ini):zero()

      -- (Rebuild the module to avoid correlated tests)
      module = nn.Sequential()
      module:add(nn.SplitTable(1))
      module:add(nn.PairwiseDistance(p))

      err = jac.testJacobian(module,input)
      mytester:assertlt(err, 1e-4, ' error on state ')

      -- Also check that the forward prop result is correct.
      -- manually calculate each distance separately
      local inputa = torch.rand(inj,ini)
      local inputb = torch.rand(inj,ini)
      local dist_manual = torch.Tensor(inj)
      for i=1, inputa:size(1) do
         dist_manual[i] = torch.dist(inputa:select(1,i), inputb:select(1,i),p)
      end
      -- compare the distances to the module's fprop
      local dist = module:forward(torch.cat(inputa,inputb,1):resize(2,inj,ini))
      err = dist - dist_manual
      mytester:assertlt(err:norm(), precision, torch.typename(module) ..
         ' error on batch fprop ')
  end
end

function nntest.Index()
    local net = nn.Index(1)

    -- test 1D
    local input = {torch.Tensor{10, 20, 30}, torch.LongTensor{1, 2, 2, 3}}
    local output = net:forward(input)
    equal(output, torch.Tensor{10, 20, 20, 30}, "error in 1D forward pass")

    local gradOutput = torch.Tensor{1, 1, 1, 3 }
    local gradInput = net:backward(input, gradOutput)
    equal(gradInput[1], torch.Tensor{1, 2, 3}, "error in 1D backward pass")

    -- test 2D
    local input = {torch.Tensor{{10, 20}, {30, 40}}, torch.LongTensor{1, 1}}
    local output = net:forward(input)
    equal(output, torch.Tensor{{10, 20}, {10, 20}}, "error in 2D forward pass")

    local gradOutput = torch.Tensor{{1, 2}, {1, 2}}
    local gradInput = net:backward(input, gradOutput)
    equal(gradInput[1], torch.Tensor{{2, 4}, {0, 0}}, "error in 2D backward pass")
end

function nntest.LookupTable()
   local totalIndex = math.random(6,9)
   local nIndex = math.random(3,5)
   local entry_size = math.random(2,5)
   local input = torch.randperm(totalIndex):narrow(1,1,nIndex):int()
   local module = nn.LookupTable(totalIndex, entry_size)
   local minval = 1
   local maxval = totalIndex

   local output = module:forward(input)
   module:backwardUpdate(input, output, 0.1)
   input:zero()

   -- 1D
   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight, minval, maxval)
   mytester:assertlt(err,precision, '1D error on weight ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight, minval, maxval)
   mytester:assertlt(err,precision, '1D error on weight [direct update] ')

   module.gradWeight:zero()
   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         '1D error on weight [%s]', t))
   end

   -- 2D
   local nframe = math.random(2,5)
   local input = torch.IntTensor(nframe, nIndex):zero()

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight, minval, maxval)
   mytester:assertlt(err,precision, '2D error on weight ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight, minval, maxval)
   mytester:assertlt(err,precision, '2D error on weight [direct update] ')

   module.gradWeight:zero()
   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         '2D error on weight [%s]', t))
   end

   -- IO
   module.gradInput = torch.Tensor(3,4):zero() --fixes an error
   local ferr,berr = jac.testIO(module,input,minval,maxval)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

   -- accUpdate
   module:accUpdateOnly()
   mytester:assert(not module.gradWeight, 'gradWeight is nil')
   module:float()
   local output = module:forward(input)
   module:backwardUpdate(input, output, 0.1)
end

function nntest.AddConstant()
  local nbatch = torch.random(3, 5)
  local f = torch.random(3, 5)
  local h = torch.random(7,9)
  local w = torch.random(7,9)
  local input = torch.rand(nbatch, f, h, w):mul(20):add(-10)  -- [-10, 10]

  local constant = torch.randn(1):squeeze()
  local mod = nn.AddConstant(constant)

  -- Test FPROP
  local output = mod:forward(input)
  local delta = output - input
  mytester:assertlt(delta:add(-constant):abs():max(), precision, 'fprop error')

  -- Test BPROP
  local err = jac.testJacobian(mod, input)
  mytester:assertlt(err, precision, 'bprop error ')

  -- inplace comparisons
  local ini = math.random(3,5)
  local inj = math.random(3,5)
  local ink = math.random(3,5)
  local constant = torch.uniform()*math.random(1,10)

  local input1 = torch.rand(ink, inj, ini)
  local input2 = input1:clone()

  local module1 = nn.AddConstant(constant,true)
  local module2 = nn.AddConstant(constant)

  local gradOutput1 = torch.rand(ink, inj, ini)
  local gradOutput2 = gradOutput1:clone()

  local out1 = module1:forward(input1)
  local out2 = module2:forward(input2)

  mytester:asserteq(0, (out1-out2):abs():max(), torch.typename(module1) ..
                    ' - in-place forward err ')

  local gradInput1 = module1:backward(input1, gradOutput1)
  local gradInput2 = module2:backward(input2, gradOutput2)

  mytester:asserteq(0, (gradInput1-gradInput2):abs():max(),
                torch.typename(module1) .. ' - in-place backward err ')

  local input1 = torch.rand(ink, inj, ini)
  local input2 = input1:clone()

  module1:forward(input1)
  module1:backward(module1.output,torch.rand(input1:size()))

  local err = (input1-input2):abs():max()
  mytester:asserteq(err, 0, torch.typename(module1) ..
                          ' - inplace input change err ')
end

function nntest.MulConstant()
  local nbatch = torch.random(3, 5)
  local f = torch.random(3, 5)
  local h = torch.random(7,9)
  local w = torch.random(7,9)
  local input = torch.rand(nbatch, f, h, w):mul(20):add(-10)  -- [-10, 10]

  local constant = torch.randn(1):squeeze()
  local mod = nn.MulConstant(constant)

  -- Test FPROP
  local output = mod:forward(input)
  local scale = output:clone():cdiv(input)
  mytester:assertlt(scale:add(-constant):abs():max(), precision, 'fprop error')

  -- Test BPROP
  local err = jac.testJacobian(mod, input)
  mytester:assertlt(err, precision, 'bprop error ')

  -- inplace comparisons
  local ini = math.random(3,5)
  local inj = math.random(3,5)
  local ink = math.random(3,5)
  local constant = torch.uniform()*math.random(1,10)

  local input1 = torch.rand(ink, inj, ini)
  local input2 = input1:clone()

  local module1 = nn.MulConstant(constant,true)
  local module2 = nn.MulConstant(constant)

  local gradOutput1 = torch.rand(ink, inj, ini)
  local gradOutput2 = gradOutput1:clone()

  local out1 = module1:forward(input1)
  local out2 = module2:forward(input2)

  mytester:asserteq(0, (out1-out2):abs():max(), torch.typename(module1) ..
                    ' - in-place forward err ')

  local gradInput1 = module1:backward(input1, gradOutput1)
  local gradInput2 = module2:backward(input2, gradOutput2)

  mytester:asserteq(0, (gradInput1-gradInput2):abs():max(),
                torch.typename(module1) .. ' - in-place backward err ')

  local input1 = torch.rand(ink, inj, ini)
  local input2 = input1:clone()

  module1:forward(input1)
  module1:backward(module1.output,torch.rand(input1:size()))

  local err = (input1-input2):abs():max()
  mytester:assertalmosteq(err, 0, 1e-15, torch.typename(module1) ..
                          ' - inplace input change err ')
end

function nntest.Copy()
   local input = torch.randn(3,4):double()
   local c = nn.Copy('torch.DoubleTensor', 'torch.FloatTensor')
   local output = c:forward(input)
   mytester:assert(torch.type(output) == 'torch.FloatTensor', 'copy forward type err')
   mytester:assertTensorEq(output, input:float(), 0.000001, 'copy forward value err')
   local gradInput = c:backward(input, output)
   mytester:assert(torch.type(gradInput) == 'torch.DoubleTensor', 'copy backward type err')
   mytester:assertTensorEq(gradInput, input, 0.000001, 'copy backward value err')
   c.dontCast = true
   c:double()
   mytester:assert(torch.type(output) == 'torch.FloatTensor', 'copy forward type err')
end

function nntest.JoinTable()
   local tensor = torch.rand(3,4,5)
   local input = {tensor, tensor}
   local module
   for d = 1,tensor:dim() do
      module = nn.JoinTable(d)
      mytester:asserteq(module:forward(input):size(d), tensor:size(d)*2, "dimension " .. d)
   end

   -- Minibatch
   local tensor = torch.rand(3,4,5)
   local input = {tensor, tensor}
   local module
   for d = 1,tensor:dim()-1 do
      module = nn.JoinTable(d, 2)
      mytester:asserteq(module:forward(input):size(d+1), tensor:size(d+1)*2, "dimension " .. d)
   end
end

function nntest.SplitTable()
   local input = torch.randn(3,4,5)
   local module
   for d = 1,input:dim() do
      module = nn.SplitTable(d)
      mytester:asserteq(#module:forward(input), input:size(d), "dimension " .. d)
   end

   -- Minibatch
   local input = torch.randn(3,4,5)
   local module
   for d = 1,input:dim()-1 do
      module = nn.SplitTable(d, 2)
      mytester:asserteq(#module:forward(input), input:size(d+1), "dimension " .. d)
   end

   -- Negative indices
   local module = nn.SplitTable(-3)
   local input = torch.randn(3,4,5)
   mytester:asserteq(#module:forward(input), 3, "negative index")
   local input = torch.randn(2,3,4,5)
   mytester:asserteq(#module:forward(input), 3, "negative index (minibatch)")
end

function nntest.SelectTable()
   local input = {
      torch.rand(3,4,5), torch.rand(3,4,5),
      {torch.rand(3,4,5)},
      {torch.rand(3,4,5), {torch.rand(3,4,5)}}
   }
   local gradOutputs = {
      torch.rand(3,4,5), torch.rand(3,4,5),
      {torch.rand(3,4,5)},
      {torch.rand(3,4,5), {torch.rand(3,4,5)}}
   }
   local zeros = {
      torch.Tensor(3,4,5):zero(), torch.Tensor(3,4,5):zero(),
      {torch.Tensor(3,4,5):zero()},
      {torch.Tensor(3,4,5):zero(), {torch.Tensor(3,4,5):zero()}}
   }
   local nonIdx = {2,3,4,1}
   local module
   for idx = 1,#input do
      module = nn.SelectTable(idx)
      local output = module:forward(input)
      equal(output, input[idx], "output dimension " .. idx)
      local gradInput = module:backward(input, gradOutputs[idx])
      equal(gradInput[idx], gradOutputs[idx], "gradInput[idx] dimension " .. idx)
      equal(gradInput[nonIdx[idx]], zeros[nonIdx[idx]], "gradInput[nonIdx] dimension " .. idx)
   end

   -- test negative index
   local idx = -2
   module = nn.SelectTable(idx)
   local output = module:forward(input)
   equal(output, input[#input+idx+1], "output dimension " .. idx)
   local gradInput = module:backward(input, gradOutputs[#input+idx+1])
   equal(gradInput[#input+idx+1], gradOutputs[#input+idx+1], "gradInput[idx] dimension " .. idx)
   equal(gradInput[nonIdx[#input+idx+1]], zeros[nonIdx[#input+idx+1]], "gradInput[nonIdx] dimension " .. idx)

   -- test typecast
   local idx = #input
   module = nn.SelectTable(idx)
   module:float()
   local output = module:forward(input)
   equal(output, input[idx], "type output")
   local gradInput = module:backward(input, gradOutputs[idx])
   equal(gradInput[idx], gradOutputs[idx], "gradInput[idx] dimension " .. idx)
   equal(gradInput[nonIdx[idx]], zeros[nonIdx[idx]], "gradInput[nonIdx] dimension " .. idx)

   -- test on differently sized sub-input tables given consequetively
   local input1 = {
      torch.rand(3,4,5),
      {torch.rand(3,4,5), torch.rand(3,4,5), torch.rand(3,4,5)}
   }
   local input2 = {
      torch.rand(3,4,5),
      {torch.rand(3,4,5), torch.rand(3,4,5)}
   }
   module = nn.SelectTable(1)
   local output = module:forward(input1)
   equal(output, input1[1], "output dimension 1")
   local gradInput = module:backward(input1, output)
   mytester:assert(#gradInput == #input1, "Table lengths")
   mytester:assert(#gradInput[2] == #input1[2], "Sub-Table lengths")
   output = module:forward(input2)
   equal(output, input2[1], "output dimension 1")
   gradInput = module:backward(input2, output)
   mytester:assert(#gradInput == #input2, "Table lengths")
   mytester:assert(#gradInput[2] == #input2[2], "Sub-Table lengths")
end

function nntest.MixtureTable()
   --[[ 2D ]]--
   -- expertInput is a Table:
   local expertInput = torch.randn(5,3,6)
   local gradOutput = torch.randn(5,6)
   local input = {
      torch.rand(5,3),
      {expertInput:select(2,1), expertInput:select(2,2), expertInput:select(2,3)}
   }
   local module = nn.MixtureTable()
   local output = module:forward(input)
   local output2 = torch.cmul(input[1]:view(5,3,1):expand(5,3,6), expertInput):sum(2)
   mytester:assertTensorEq(output, output2, 0.000001, "mixture output")
   local gradInput = module:backward(input, gradOutput)
   local gradOutput2 = torch.view(gradOutput, 5, 1, 6):expandAs(expertInput)
   local gaterGradInput2 = torch.cmul(gradOutput2, expertInput):sum(3):select(3,1)
   mytester:assertTensorEq(gradInput[1], gaterGradInput2, 0.000001, "mixture gater gradInput")
   local expertGradInput2 = torch.cmul(input[1]:view(5,3,1):expand(5,3,6), gradOutput:view(5,1,6):expand(5,3,6))
   for i, expertGradInput in ipairs(gradInput[2]) do
      mytester:assertTensorEq(expertGradInput, expertGradInput2:select(2,i), 0.000001, "mixture expert "..i.." gradInput")
   end
   -- expertInput is a Tensor:
   local input = {input[1], expertInput}
   local module = nn.MixtureTable(2)
   local output = module:forward(input)
   mytester:assertTensorEq(output, output2, 0.000001, "mixture2 output")
   local gradInput = module:backward(input, gradOutput)
   mytester:assertTensorEq(gradInput[1], gaterGradInput2, 0.000001, "mixture2 gater gradInput")
   mytester:assertTensorEq(gradInput[2], expertGradInput2, 0.000001, "mixture2 expert gradInput")

   --[[ 3D ]]--
   local expertInput = torch.randn(5,6,3,2)
   local gradOutput = torch.randn(5,6,2)
   -- expertInput is a Table:
   local input = {
      torch.rand(5,3),
      {expertInput:select(3,1), expertInput:select(3,2), expertInput:select(3,3)}
   }
   local module = nn.MixtureTable()
   local output = module:forward(input)
   local output2 = torch.cmul(input[1]:view(5,1,3,1):expand(5,6,3,2), expertInput):sum(3)
   mytester:assertTensorEq(output, output2, 0.000001, "mixture3 output")
   local gradInput = module:backward(input, gradOutput)
   local gradOutput2 = torch.view(gradOutput,5,6,1,2):expandAs(expertInput)
   local gaterGradInput2 = torch.cmul(gradOutput2, expertInput):sum(4):select(4,1):sum(2):select(2,1)
   mytester:assertTensorEq(gradInput[1], gaterGradInput2, 0.000001, "mixture3 gater gradInput")
   local expertGradInput2 = torch.cmul(input[1]:view(5,1,3,1):expand(5,6,3,2), gradOutput2)
   for i, expertGradInput in ipairs(gradInput[2]) do
      mytester:assertTensorEq(expertGradInput, expertGradInput2:select(3,i), 0.000001, "mixture3 expert "..i.." gradInput")
   end
   -- expertInput is a Tensor
   local input = {input[1], expertInput}
   local module = nn.MixtureTable(3)
   local output = module:forward(input)
   mytester:assertTensorEq(output, output2, 0.000001, "mixture4 output")
   local gradInput = module:backward(input, gradOutput)
   mytester:assertTensorEq(gradInput[1], gaterGradInput2, 0.000001, "mixture4 gater gradInput")
   mytester:assertTensorEq(gradInput[2], expertGradInput2, 0.000001, "mixture4 expert gradInput")

   --[[ 1D ]]--
   -- expertInput is a Table:
   local expertInput = torch.randn(3,6)
   local gradOutput = torch.randn(6)
   local input = {
      torch.rand(3),
      {expertInput:select(1,1), expertInput:select(1,2), expertInput:select(1,3)}
   }
   local module = nn.MixtureTable()
   local output = module:forward(input)
   local output2 = torch.cmul(input[1]:view(3,1):expand(3,6), expertInput):sum(1)
   mytester:assertTensorEq(output, output2, 0.000001, "mixture5 output")
   local gradInput = module:backward(input, gradOutput)
   local gradOutput2 = torch.view(gradOutput, 1, 6):expandAs(expertInput)
   local gaterGradInput2 = torch.cmul(gradOutput2, expertInput):sum(2):select(2,1)
   mytester:assertTensorEq(gradInput[1], gaterGradInput2, 0.000001, "mixture5 gater gradInput")
   local expertGradInput2 = torch.cmul(input[1]:view(3,1):expand(3,6), gradOutput:view(1,6):expand(3,6))
   for i, expertGradInput in ipairs(gradInput[2]) do
      mytester:assertTensorEq(expertGradInput, expertGradInput2:select(1,i), 0.000001, "mixture5 expert "..i.." gradInput")
   end
   -- test type-cast
   module:float()
   local input2 = {
      input[1]:float(),
      {input[2][1]:float(), input[2][2]:float(), input[2][3]:float()}
   }
   local output = module:forward(input2)
   mytester:assertTensorEq(output, output2:float(), 0.000001, "mixture5B output")
   local gradInput = module:backward(input2, gradOutput:float())
   mytester:assertTensorEq(gradInput[1], gaterGradInput2:float(), 0.000001, "mixture5B gater gradInput")
   for i, expertGradInput in ipairs(gradInput[2]) do
      mytester:assertTensorEq(expertGradInput, expertGradInput2:select(1,i):float(), 0.000001, "mixture5B expert "..i.." gradInput")
   end
   -- expertInput is a Tensor:
   local input = {input[1], expertInput}
   local module = nn.MixtureTable(1)
   local output = module:forward(input)
   mytester:assertTensorEq(output, output2, 0.000001, "mixture6 output")
   local gradInput = module:backward(input, gradOutput)
   mytester:assertTensorEq(gradInput[1], gaterGradInput2, 0.000001, "mixture6 gater gradInput")
   mytester:assertTensorEq(gradInput[2], expertGradInput2, 0.000001, "mixture6 expert gradInput")
   -- test type-cast:
   module:float()
   local input2 = {input[1]:float(), expertInput:float()}
   local output = module:forward(input2)
   mytester:assertTensorEq(output, output2:float(), 0.000001, "mixture6B output")
   local gradInput = module:backward(input2, gradOutput:float())
   mytester:assertTensorEq(gradInput[1], gaterGradInput2:float(), 0.000001, "mixture6B gater gradInput")
   mytester:assertTensorEq(gradInput[2], expertGradInput2:float(), 0.000001, "mixture6B expert gradInput")

   --[[ 2D gater, 1D expert]]--
   -- expertInput is a Table:
   local expertInput = torch.randn(5,3)
   local gradOutput = torch.randn(5)
   local input = {
      torch.rand(5,3),
      {expertInput:select(2,1), expertInput:select(2,2), expertInput:select(2,3)}
   }
   local module = nn.MixtureTable()
   local output = module:forward(input)
   local output2 = torch.cmul(input[1], expertInput):sum(2)
   mytester:assertTensorEq(output, output2, 0.000001, "mixture7 output")
   local gradInput = module:backward(input, gradOutput)
   local gradOutput2 = torch.view(gradOutput, 5, 1):expandAs(expertInput)
   local gaterGradInput2 = torch.cmul(gradOutput2, expertInput)
   mytester:assertTensorEq(gradInput[1], gaterGradInput2, 0.000001, "mixture7 gater gradInput")
   local expertGradInput2 = torch.cmul(input[1], gradOutput:view(5,1):expand(5,3))
   for i, expertGradInput in ipairs(gradInput[2]) do
      mytester:assertTensorEq(expertGradInput, expertGradInput2:select(2,i), 0.000001, "mixture7 expert "..i.." gradInput")
   end
end

function nntest.NarrowTable()
   local input = torch.randn(3,10,4)
   local gradOutput = torch.randn(3,3,4)
   local nt = nn.NarrowTable(5,3)
   local seq = nn.Sequential()
   seq:add(nn.SplitTable(1,2))
   seq:add(nt)
   seq:add(nn.JoinTable(1,1))
   seq:add(nn.Reshape(3,3,4))
   local seq2 = nn.Narrow(2,5,3)
   local output = seq:forward(input)
   local gradInput = seq:backward(input, gradOutput)
   local output2 = seq2:forward(input)
   local gradInput2 = seq2:backward(input, gradOutput)
   mytester:assertTensorEq(output, output2, 0.0000001, "NarrowTable output err")
   mytester:assertTensorEq(gradInput, gradInput2, 0.00001, "NarrowTable gradInput err")

   -- now try it with a smaller input
   local input = input:narrow(2, 1, 8)
   local output = seq:forward(input)
   local gradInput = seq:backward(input, gradOutput)
   local output2 = seq2:forward(input)
   local gradInput2 = seq2:backward(input, gradOutput)
   mytester:assertTensorEq(output, output2, 0.0000001, "NarrowTable small output err")
   mytester:assertTensorEq(gradInput, gradInput2, 0.00001, "NarrowTable small gradInput err")

   -- test type-cast
   local input = input:float()
   local gradOutput = gradOutput:float()
   seq:float()
   seq2:float()
   local output = seq:forward(input)
   local gradInput = seq:backward(input, gradOutput)
   local output2 = seq2:forward(input)
   local gradInput2 = seq2:backward(input, gradOutput)
   mytester:assertTensorEq(output, output2, 0.0000001, "NarrowTable output float err")
   mytester:assertTensorEq(gradInput, gradInput2, 0.00001, "NarrowTable gradInput float err")
end

function nntest.View()
   local input = torch.rand(10)
   local template = torch.rand(5,2)
   local target = template:size():totable()
   local module = nn.View(template:size())
   mytester:assertTableEq(module:forward(input):size():totable(), target, "Error in forward (1)")
   local module = nn.View(table.unpack(target))
   mytester:assertTableEq(module:forward(input):size():totable(), target, "Error in forward (2)")

   -- Minibatch
   local minibatch = torch.rand(5,10)
   mytester:assertTableEq(module:forward(minibatch):size(1),
      minibatch:size(1),
      "Error in minibatch dimension")
   mytester:assertTableEq(module:forward(minibatch):nElement(),
      minibatch:nElement(),
      "Error in minibatch nElement")
   local module = nn.View(-1):setNumInputDims(1)
   mytester:assertTableEq(module:forward(minibatch):size(1),
      minibatch:size(1),
      "Error in minibatch dimension with size -1")
   mytester:assertTableEq(module:forward(minibatch):nElement(),
      minibatch:nElement(),
      "Error in minibatch nElement with size -1")

   -- another setNumInputDims case
   local minibatch = torch.rand(5,4,10)
   local module = nn.View(-1):setNumInputDims(2)
   mytester:assertTableEq(module:forward(minibatch):size(1),
      minibatch:size(1),
      "Error in minibatch dimension with size -1")

   -- another setNumInputDims case
   local minibatch = torch.rand(2,5,4,10)
   local module = nn.View(4,-1):setNumInputDims(2)
   local out = module:forward(minibatch)
   mytester:assertTableEq(out:size(1), minibatch:size(1)*minibatch:size(2),
                          "Error in minibatch dimension with size -1")
   mytester:assertTableEq(out:size(2), minibatch:size(3),
                          "Error in minibatch dimension with size -1")
   mytester:assertTableEq(out:size(3), minibatch:size(4),
                          "Error in minibatch dimension with size -1")

   -- Minibatch Generalization
   local minibatch = torch.rand(5,2,6)
   local module = nn.View(6)
   mytester:assertTableEq(
      module:forward(minibatch):size(1),
      minibatch:size(1)*minibatch:size(2),
      "Error in minibatch generalization dimension")
   mytester:assertTableEq(
      module:forward(minibatch):nElement(),
      minibatch:nElement(),
      "Error in minibatch generalization nElement")
end

function nntest.Reshape()
   local input = torch.rand(10)
   local template = torch.rand(5,2)
   local target = template:size():totable()
   local module = nn.Reshape(template:size())
   mytester:assertTableEq(module:forward(input):size():totable(), target, "Error in forward (1)")
   local module = nn.View(table.unpack(target))
   mytester:assertTableEq(module:forward(input):size():totable(), target, "Error in forward (2)")

   -- Minibatch
   local minibatch = torch.rand(5,10)
   mytester:assertTableEq(module:forward(minibatch):size(1),
      minibatch:size(1),
      "Error in minibatch dimension")
   mytester:assertTableEq(module:forward(minibatch):nElement(),
      minibatch:nElement(),
      "Error in minibatch nElement")
end

-- Define a test for SpatialUpSamplingCuda
function nntest.SpatialUpSamplingNearest()
  local scale = torch.random(2,4)
  for dim = 3,4 do
    local m = nn.SpatialUpSamplingNearest(scale)

    -- Create a randomly sized dimD vector
    local shape = {}
    for i = 1, dim do
      table.insert(shape, torch.random(2, 2+dim-1))
    end

    -- Check that the gradient is correct by using finite elements
    local input = torch.Tensor(table.unpack(shape)):zero()

    local err = jac.testJacobian(m, input)
    mytester:assertlt(err, precision, ' error on state ')

    local ferr, berr = jac.testIO(m, input)
    mytester:asserteq(ferr, 0, torch.typename(m)..' - i/o forward err ')
    mytester:asserteq(berr, 0, torch.typename(m)..' - i/o backward err ')
  end
end

function nntest.Parallel()
   local input = torch.randn(3, 4, 5)
   local m = nn.Parallel(1,3)
   m:add(nn.View(4,5,1))
   m:add(nn.View(4,5,1))
   m:add(nn.View(4,5,1))

   local output = m:forward(input)
   local output2 = input:transpose(1,3):transpose(1,2)
   mytester:assertTensorEq(output2, output, 0.000001, 'Parallel forward err')

   local gradInput = m:backward(input, output2)
   mytester:assertTensorEq(gradInput, input, 0.000001, 'Parallel backward err')
end

function nntest.ParallelTable()
   local input = torch.randn(3, 4, 5)
   local p = nn.ParallelTable()
   p:add(nn.View(4,5,1))
   p:add(nn.View(4,5,1))
   p:add(nn.View(4,5,1))
   local m = nn.Sequential()
   m:add(nn.SplitTable(1))
   m:add(p)
   m:add(nn.JoinTable(3))

   local output = m:forward(input)
   local output2 = input:transpose(1,3):transpose(1,2)
   mytester:assertTensorEq(output2, output, 0.000001, 'ParallelTable forward err')

   local gradInput = m:backward(input, output2)
   mytester:assertTensorEq(gradInput, input, 0.000001, 'ParallelTable backward err')
end

function nntest.ConcatTable()
   -- Test tensor input
   local input = torch.rand(5, 5, 5)
   local m = nn.Sequential()

   local concat = nn.ConcatTable()
   concat:add(nn.Identity())

   m:add(concat)  -- Output of concat is a table of length 1
   m:add(nn.JoinTable(1))  -- jac needs a tensor tensor output

   local err = jac.testJacobian(m, input)
   mytester:assertlt(err, precision, ' error on state ')

   local ferr, berr = jac.testIO(m, input)
   mytester:asserteq(ferr, 0, torch.typename(m)..' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(m)..' - i/o backward err ')

   -- Now test a table input
   local input = {
      torch.randn(3,4):float(), torch.randn(3,4):float(), {torch.randn(3,4):float()}
   }
   local _gradOutput = {
      torch.randn(3,3,4):float(), torch.randn(3,3,4):float(), torch.randn(3,3,4):float()
   }
   local gradOutput = {
      {_gradOutput[1][1], _gradOutput[2][1], {_gradOutput[3][1]}},
      {_gradOutput[1][2], _gradOutput[2][2], {_gradOutput[3][2]}},
      {_gradOutput[1][3], _gradOutput[2][3], {_gradOutput[3][3]}}
   }
   local module = nn.ConcatTable()
   module:add(nn.Identity())
   module:add(nn.Identity())
   module:add(nn.Identity())
   module:float()

   local output = module:forward(input)
   local output2 = {input, input, input}
   equal(output2, output, "ConcatTable table output")
   local gradInput = module:backward(input, gradOutput)
   local gradInput2 = {_gradOutput[1]:sum(1), _gradOutput[2]:sum(1), {_gradOutput[3]:sum(1)}}
   equal(gradInput, gradInput2, "ConcatTable table gradInput")

   -- test outputs for variable length inputs
   local test = nn.ConcatTable()
   test:add(nn.Identity())
   test:add(nn.Identity())

   local x = {torch.randn(5), torch.randn(5)}
   local y = {torch.randn(5)}

   local o1 = #(test:forward(x))
   local go1 = #(test:backward(x, {x, x}))
   local o2 = #(test:forward(y))
   local go2 = #(test:backward(y, {y, y}))
   mytester:assert(o1 == 2, "ConcatTable table variable length")
   mytester:assert(go1 == 2, "ConcatTable table variable length")
   mytester:assert(o2 == 2, "ConcatTable table variable length")
   mytester:assert(go2 == 1, "ConcatTable table variable length")
end

function nntest.FlattenTable()
   -- Create a nested table.  Obviously we can't even stochastically test
   -- the space of all possible nested tables (it's infinite), but here is a
   -- hand-coded one that covers all the cases we need:
   local input = {
     torch.rand(1),
     {
       torch.rand(2),
       {
         torch.rand(3)
       },
     },
     torch.rand(4)
   }
   local gradOutput = {
     torch.rand(1),
     torch.rand(2),
     torch.rand(3),
     torch.rand(4)
   }

   -- Check the FPROP
   local m = nn.FlattenTable()
   local output = m:forward(input)
   mytester:assert(#output == 4, torch.typename(m)..' - fprop err ')
   -- This is ugly, but check that the mapping from input to output is correct
   mytester:assert(output[1] == input[1])
   mytester:assert(output[2] == input[2][1])
   mytester:assert(output[3] == input[2][2][1])
   mytester:assert(output[4] == input[3])

   -- Check the BPROP
   local gradInput = m:backward(input, gradOutput)
   -- Again, check that the mapping is correct
   mytester:assert(gradOutput[1] == gradInput[1])
   mytester:assert(gradOutput[2] == gradInput[2][1])
   mytester:assert(gradOutput[3] == gradInput[2][2][1])
   mytester:assert(gradOutput[4] == gradInput[3])

   -- More uglyness: FlattenTable doesn't rebuild the table every updateOutput
   -- call, so we need to make sure that modifications to the input are
   -- detected correctly (and that the table is correctly rebuilt.
   -- CASE 1: Nothing changes so the output table shouldn't be redefined
   local old_input_map = m.input_map
   local old_output = m.output
   local _ = m:forward(input)
   mytester:assert(old_input_map == m.input_map and old_output == m.output)

   -- CASE 2: An element is added to the input table
   old_input_map = m.input_map
   old_output = m.output
   input[2][#(input[2])+1] = torch.rand(5)
   m:forward(input)
   mytester:assert(old_input_map ~= m.input_map and old_output ~= m.output)

   -- CASE 3: An element is removed from the input table
   old_input_map = m.input_map
   old_output = m.output
   input[#input] = nil
   m:forward(input)
   mytester:assert(old_input_map ~= m.input_map and old_output ~= m.output)

   -- At this point further testing is not necessary I think, but just to be
   -- consistent: perform a jacobian test by using SplitTable and JointTable
   -- elements
   m = nn.Sequential()
   local par = nn.ParallelTable()
   par:add(nn.SplitTable(1))
   par:add(nn.SplitTable(1))
   m:add(nn.SplitTable(1))
   m:add(par)  -- this will create a nested table
   m:add(nn.FlattenTable())  -- This will flatten the nested table
   m:add(nn.JoinTable(1))  -- Finally, this will create a 1D tensor

   input = torch.Tensor(2,2,2)
   local err = jac.testJacobian(m, input)
   mytester:assertlt(err, precision, 'error on bprop ')
end

function nntest.L1Penalty()
   local weight = 1
   local sizeAverage = false
   local m = nn.L1Penalty(weight, sizeAverage, false)

   local input = torch.rand(2,10):add(-0.5)
   input[1][1] = 0

   local _ = m:forward(input)
   local grad = m:backward(input, torch.ones(input:size()))

   local err = input:clone():abs():sum()*weight - m.loss
   mytester:assertlt(math.abs(err), precision, 'error on fprop ')

   local true_grad = (input:gt(0):typeAs(grad) +
      input:lt(0):typeAs(grad):mul(-1)):mul(weight)
   mytester:assertlt((true_grad - grad):abs():max(), precision,
      'error on bprop ')

   -- Note: We cannot use the Jacobian test for this Module since the backward
   -- gradient cannot be estimated using finite differences (ie, the loss
   -- during BPROP is not included in the FPROP output)
end

function nntest.L1Cost()
   local input = torch.rand(10) * 2 - 1
   local m = nn.L1Cost()
   local output = m:forward(input)
   local err = output - torch.abs(input):sum()
   mytester:assertalmosteq(err, 0, 1e-15, 'L1Cost forward')
end

function nntest.DepthConcat()
   local outputSize = torch.IntTensor{5,6,7,8}
   local input = torch.randn(2,3,12,12)
   local gradOutput = torch.randn(2, outputSize:sum(), 12, 12)
   local concat = nn.DepthConcat(2)
   concat:add(nn.SpatialConvolutionMM(3, outputSize[1], 1, 1, 1, 1)) --> 2, 5, 12, 12
   concat:add(nn.SpatialConvolutionMM(3, outputSize[2], 3, 3, 1, 1)) --> 2, 6, 10, 10
   concat:add(nn.SpatialConvolutionMM(3, outputSize[3], 4, 4, 1, 1)) --> 2, 7, 9, 9
   concat:add(nn.SpatialConvolutionMM(3, outputSize[4], 5, 5, 1, 1)) --> 2, 8, 8, 8
   concat:zeroGradParameters()
   -- forward/backward
   local outputConcat = concat:forward(input)
   local gradInputConcat = concat:backward(input, gradOutput)
   -- the spatial dims are the largest, the nFilters is the sum
   local output = torch.Tensor(2, outputSize:sum(), 12, 12):zero() -- zero for padding
   local narrows = { {{},{1,5},{},{}}, {{},{6,11},{2,11},{2,11}}, {{},{12,18},{2,10},{2,10}}, {{},{19,26},{3,10},{3,10}} }
   local gradInput = input:clone():zero()
   for i=1,4 do
      local conv = concat:get(i)
      local gradWeight = conv.gradWeight:clone()
      conv:zeroGradParameters()
      output[narrows[i]]:copy(conv:forward(input))
      gradInput:add(conv:backward(input, gradOutput[narrows[i]]))
      mytester:assertTensorEq(gradWeight, conv.gradWeight, 0.000001, "Error in SpatialConcat:accGradParameters for conv "..i)
   end
   mytester:assertTensorEq(output, outputConcat, 0.000001, "Error in SpatialConcat:updateOutput")
   mytester:assertTensorEq(gradInput, gradInputConcat, 0.000001, "Error in SpatialConcat:updateGradInput")
end

local function createMatrixInputSizes()
  local M = torch.random(10, 20)
  local N = torch.random(10, 20)
  local P = torch.random(10, 20)
  return M, N, P
end

function nntest.MM()
  local mm = nn.MM(false, true)
  local M, N, P = createMatrixInputSizes()
  local A = torch.randn(M, N)
  local B = torch.randn(P, N)

  -- Test forward pass.
  local output = mm:forward({A, B})
  mytester:assertTableEq(output:size():totable(), {M, P},
                         'Output has wrong dimensionality')
  mytester:assertTensorEq(output, A * B:t(), 1e-10,
                          'Wrong output')

  -- Test backward pass.
  local gradOutput = torch.randn(M, P)
  local gradInput = mm:backward({A, B}, gradOutput)
  mytester:assert(#gradInput == 2, 'gradInput must be table of size 2')
  local gradA, gradB = table.unpack(gradInput)
  mytester:assertTableEq(gradA:size():totable(), A:size():totable(),
                         'Gradient for input A has wrong size')
  mytester:assertTableEq(gradB:size():totable(), B:size():totable(),
                         'Gradient for input B has wrong size')
  mytester:assertTensorEq(gradA, gradOutput * B, 1e-10,
                          'Wrong gradient for input A')
  mytester:assertTensorEq(gradB, gradOutput:t() * A, 1e-10,
                          'Wrong gradient for input B')
end

function nntest.BatchMMNoTranspose()
  local mm = nn.MM()
  local M, N, P = createMatrixInputSizes()
  for bSize = 1, 11, 5 do
    local A = torch.randn(bSize, M, N)
    local B = torch.randn(bSize, N, P)

    -- Test forward pass.
    local output = mm:forward({A, B})
    mytester:assertTableEq(output:size():totable(), {bSize, M, P},
                           'Output has wrong dimensionality')
    for i = 1, bSize do
      mytester:assertTensorEq(output[i], A[i] * B[i], 1e-10,
                              'Output wrong for bSize = ' .. bSize .. ' and i = ' .. i)
    end

    -- Test backward pass.
    local gradOutput = torch.randn(bSize, M, P)
    local gradInput = mm:backward({A, B}, gradOutput)
    mytester:assert(#gradInput == 2, 'gradInput must be table of size 2')
    local gradA, gradB = table.unpack(gradInput)
    mytester:assertTableEq(gradA:size():totable(), A:size():totable(),
                           'Gradient for input A has wrong size')
    mytester:assertTableEq(gradB:size():totable(), B:size():totable(),
                           'Gradient for input B has wrong size')
    for i = 1, bSize do
      mytester:assertTensorEq(gradA[i], gradOutput[i] * B[i]:t(), 1e-10,
                              'Gradient for input A wrong for bSize = ' .. bSize .. ' and i = ' .. i)
      mytester:assertTensorEq(gradB[i], A[i]:t() * gradOutput[i], 1e-10,
                              'Gradient for input B wrong for bSize = ' .. bSize .. ' and i = ' .. i)
    end
  end
end

function nntest.BatchMMTransposeA()
  local mm = nn.MM(true, false)
  local M, N, P = createMatrixInputSizes()
  for bSize = 1, 11, 5 do
    local A = torch.randn(bSize, N, M)
    local B = torch.randn(bSize, N, P)

    -- Test forward pass.
    local output = mm:forward({A, B})
    mytester:assertTableEq(output:size():totable(), {bSize, M, P},
                           'Output has wrong dimensionality')
    for i = 1, bSize do
      mytester:assertTensorEq(output[i], A[i]:t() * B[i], 1e-10,
                              'Output wrong for bSize = ' .. bSize .. ' and i = ' .. i)
    end

    -- Test backward pass.
    local gradOutput = torch.randn(bSize, M, P)
    local gradInput = mm:backward({A, B}, gradOutput)
    mytester:assert(#gradInput == 2, 'gradInput must be table of size 2')
    local gradA, gradB = table.unpack(gradInput)
    mytester:assertTableEq(gradA:size():totable(), A:size():totable(),
                           'Gradient for input A has wrong size')
    mytester:assertTableEq(gradB:size():totable(), B:size():totable(),
                           'Gradient for input B has wrong size')
    for i = 1, bSize do
      mytester:assertTensorEq(gradA[i], B[i] * gradOutput[i]:t(), 1e-10,
                              'Gradient for input A wrong for bSize = ' .. bSize .. ' and i = ' .. i)
      mytester:assertTensorEq(gradB[i], A[i] * gradOutput[i], 1e-10,
                              'Gradient for input B wrong for bSize = ' .. bSize .. ' and i = ' .. i)
    end
  end
end

function nntest.BatchMMTransposeB()
  local mm = nn.MM(false, true)
  local M, N, P = createMatrixInputSizes()
  for bSize = 1, 11, 5 do
    local A = torch.randn(bSize, M, N)
    local B = torch.randn(bSize, P, N)

    -- Test forward pass.
    local output = mm:forward({A, B})
    mytester:assertTableEq(output:size():totable(), {bSize, M, P},
                           'Output has wrong dimensionality')
    for i = 1, bSize do
      mytester:assertTensorEq(output[i], A[i] * B[i]:t(), 1e-10,
                              'Output wrong for bSize = ' .. bSize .. ' and i = ' .. i)
    end

    -- Test backward pass.
    local gradOutput = torch.randn(bSize, M, P)
    local gradInput = mm:backward({A, B}, gradOutput)
    mytester:assert(#gradInput == 2, 'gradInput must be table of size 2')
    local gradA, gradB = table.unpack(gradInput)
    mytester:assertTableEq(gradA:size():totable(), A:size():totable(),
                           'Gradient for input A has wrong size')
    mytester:assertTableEq(gradB:size():totable(), B:size():totable(),
                           'Gradient for input B has wrong size')
    for i = 1, bSize do
      mytester:assertTensorEq(gradA[i], gradOutput[i] * B[i], 1e-10,
                              'Gradient for input A wrong for bSize = ' .. bSize .. ' and i = ' .. i)
      mytester:assertTensorEq(gradB[i], gradOutput[i]:t() * A[i], 1e-10,
                              'Gradient for input B wrong for bSize = ' .. bSize .. ' and i = ' .. i)
    end
  end
end

function nntest.BatchMMTransposeBoth()
  local mm = nn.MM(true, true)
  local M, N, P = createMatrixInputSizes()
  for bSize = 1, 11, 5 do
    local A = torch.randn(bSize, N, M)
    local B = torch.randn(bSize, P, N)

    -- Test forward pass.
    local output = mm:forward({A, B})
    mytester:assertTableEq(output:size():totable(), {bSize, M, P},
                           'Output has wrong dimensionality')
    for i = 1, bSize do
      mytester:assertTensorEq(output[i], A[i]:t() * B[i]:t(), 1e-10,
                              'Output wrong for bSize = ' .. bSize .. ' and i = ' .. i)
    end

    -- Test backward pass.
    local gradOutput = torch.randn(bSize, M, P)
    local gradInput = mm:backward({A, B}, gradOutput)
    mytester:assert(#gradInput == 2, 'gradInput must be table of size 2')
    local gradA, gradB = table.unpack(gradInput)
    mytester:assertTableEq(gradA:size():totable(), A:size():totable(),
                           'Gradient for input A has wrong size')
    mytester:assertTableEq(gradB:size():totable(), B:size():totable(),
                           'Gradient for input B has wrong size')
    for i = 1, bSize do
      mytester:assertTensorEq(gradA[i], B[i]:t() * gradOutput[i]:t(), 1e-10,
                              'Gradient for input A wrong for bSize = ' .. bSize .. ' and i = ' .. i)
      mytester:assertTensorEq(gradB[i], gradOutput[i]:t() * A[i]:t(), 1e-10,
                              'Gradient for input B wrong for bSize = ' .. bSize .. ' and i = ' .. i)
    end
  end
end

function nntest.DotProduct()
  local indim = math.random(1,10)

  -- test 1D forward
  local input = {torch.rand(indim),torch.rand(indim)}
  local module = nn.DotProduct()
  local expected = input[1]:dot(input[2])
  local output = module:forward(input)
  mytester:assertlt(math.abs(expected-output[1]), precision, 'error on forward ')

  -- check gradients
  -- Note: testJacobian doesn't support table inputs, and rather than re-write
  -- it so that it does, I'll just use a split table module on the input.
  -- I assume both SplitTable and Sequential do not have bugs, otherwise this
  -- test will break.
  local input = torch.rand(2,indim)
  local module = nn.Sequential()
  module:add(nn.SplitTable(1))
  module:add(nn.DotProduct())

  local err = jac.testJacobian(module,input)
  mytester:assertlt(err,precision, 'error on state ')

  -- IO
  local ferr,berr = jac.testIO(module,input)
  mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
  mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

  -- batch
  -- rebuild module to avoid correlated tests
  local module = nn.Sequential()
  module:add(nn.SplitTable(1))
  module:add(nn.DotProduct())

  local nframes = math.random(1,10)
  local indim = math.random(1,10)
  local input = torch.rand(2,nframes,indim)

  local err = jac.testJacobian(module,input)
  mytester:assertlt(err,precision, 'batch error on state ')
end

function nntest.CosineDistance()
  local indim = math.random(1,10)
  local input = {torch.rand(indim),torch.rand(indim)}

  -- check forward against previous implementation
  local module = nn.CosineDistance()

  local w1 = input[1]:dot(input[2])
  local w2 = math.sqrt(input[1]:dot(input[1]))
  local w3 = math.sqrt(input[2]:dot(input[2]))
  local output_old = w1/w2/w3

  local output = module:forward(input)

  mytester:assertlt(math.abs(output_old-output[1]),precision,'error on forward ')


  -- check gradients
  -- Note: testJacobian doesn't support table inputs, and rather than re-write
  -- it so that it does, I'll just use a split table module on the input.
  -- I assume both SplitTable and Sequential do not have bugs, otherwise this
  -- test will break.
  local input = torch.rand(2,indim)
  local module = nn.Sequential()
  module:add(nn.SplitTable(1))
  module:add(nn.CosineDistance())

  local err = jac.testJacobian(module,input)
  mytester:assertlt(err,precision, 'error on state ')

  -- IO
  local ferr,berr = jac.testIO(module,input)
  mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
  mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

  -- batch
  -- rebuild module to avoid correlated tests
  local module = nn.Sequential()
  module:add(nn.SplitTable(1))
  module:add(nn.CosineDistance())

  local nframes = math.random(1,10)
  local indim = math.random(1,10)
  local input = torch.rand(2,nframes,indim)

  local err = jac.testJacobian(module,input)
  mytester:assertlt(err,precision, 'batch error on state ')

end

function nntest.CosineEmbeddingCriterion()
  local v1 = torch.Tensor{1, 0}
  local v2 = torch.Tensor{0.5, math.sqrt(3)*0.5}

  local crit = nn.CosineEmbeddingCriterion(0.6)
  local output = crit:forward({v1, v2}, -1) -- must be Called before backward
  local grads = crit:backward({v1, v2}, -1)

  local zero = torch.Tensor(2):zero()
  equal(grads[1], zero, 'gradient should be zero')
  equal(grads[2], zero, 'gradient should be zero')

  -- check jacobians
  local margin = math.random()*2-1
  local dim = 5
  local batch_size = 1
  local crit = nn.CosineEmbeddingCriterion(margin)
  local v = torch.rand(2,dim)
  criterionJacobianTest1DTable(crit,v,1)
  criterionJacobianTest1DTable(crit,v,-1)

  -- batch with hand-computed values
  local v1 = torch.Tensor{{1, 0}, {0.5, math.sqrt(3)*0.5}}
  local v2 = torch.Tensor{{0.5, math.sqrt(3)*0.5}, {1, 0}}

  local t = torch.Tensor{-1,-1}
  local crit = nn.CosineEmbeddingCriterion(0.6)
  local output = crit:forward({v1, v2}, t) -- must be Called before backward
  local grads = crit:backward({v1, v2}, t)

  local zero = torch.Tensor(2,2):zero()
  equal(grads[1], zero, 'gradient should be zero')
  equal(grads[2], zero, 'gradient should be zero')

  -- batch, sizeAverage true, jacobian
  local margin = math.random()*2-1
  local dim = 5
  local batch_size = 2
  local crit = nn.CosineEmbeddingCriterion(margin)
  crit.sizeAverage = true
  local v = torch.rand(2,batch_size,dim)
  local t = torch.Tensor(batch_size):random(0,1):mul(2):add(-1)
  criterionJacobianTest1DTable(crit,v,t)

  -- batch, sizeAverage false, jacobian
  local margin = math.random()*2-1
  local crit = nn.CosineEmbeddingCriterion(margin)
  crit.sizeAverage = false
  local v = torch.rand(2,batch_size,dim)
  local t = torch.Tensor(batch_size):random(0,1):mul(2):add(-1)
  criterionJacobianTest1DTable(crit,v,t)
end

function nntest.HingeEmbeddingCriterion()
  local x = torch.Tensor{0.3,2.1,1.8,0}
  local y = torch.Tensor{1,-1,-1,1}
  local expgrads = torch.Tensor{1,0,-1,1} / 4

  local crit = nn.HingeEmbeddingCriterion(2)
  local output = crit:forward(x, y) -- must be called before backward
  local grads = crit:backward(x, y)

  mytester:assert(math.abs(output - (0.3 + 0.2) / 4) < 1e-10)
  equal(grads, expgrads)
end

function nntest.Replicate()
   local vector = torch.rand(3)

   local r1 = nn.Replicate(2, 1)
   local r2 = nn.Replicate(2, 2)

   local vOutput1 = r1:forward(vector):clone()
   local vOutput2 = r2:forward(vector):clone()

   local expected1 = torch.zeros(2, 3)
   local expected2 = torch.zeros(3, 2)
   expected1:select(1, 1):copy(vector)
   expected1:select(1, 2):copy(vector)
   expected2:select(2, 1):copy(vector)
   expected2:select(2, 2):copy(vector)

   mytester:assertTensorEq(vOutput1, expected1, precision, 'Wrong tiling of data when replicating vector.')
   mytester:assertTensorEq(vOutput2, expected2, precision, 'Wrong tiling of data when replicating vector.')

   -- batch mode
   local vector = torch.rand(4,3)

   local r1 = nn.Replicate(2, 1, 1)
   local r2 = nn.Replicate(2, 2, 1)

   local vOutput1 = r1:forward(vector):clone()
   local vOutput2 = r2:forward(vector):clone()

   local expected1 = torch.zeros(4, 2, 3)
   local expected2 = torch.zeros(4, 3, 2)
   expected1:select(2, 1):copy(vector)
   expected1:select(2, 2):copy(vector)
   expected2:select(3, 1):copy(vector)
   expected2:select(3, 2):copy(vector)

   mytester:assertTensorEq(vOutput1, expected1, precision, 'Wrong tiling of data when replicating batch vector.')
   mytester:assertTensorEq(vOutput2, expected2, precision, 'Wrong tiling of data when replicating batch vector.')
end

function nntest.BatchNormalization()
   local nframes = torch.random(50,70)
   local indim = torch.random(1,10)
   local input = torch.zeros(nframes, indim):uniform()
   local module = nn.BatchNormalization(indim)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input,
                                      module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input,
                                      module.bias, module.gradBias)
   mytester:assertlt(err,precision, 'error on weight ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err,precision, 'error on weight [direct update] ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err,precision, 'error on bias [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input,
                                        'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                           'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input,
                                        'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                           'error on bias [%s]', t))
   end

   -- IO
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

   -- batch norm without affine transform
   module = nn.BatchNormalization(indim, 1e-5, 0.1, false)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   -- IO
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SpatialBatchNormalization()
   local nframes = torch.random(1,10)
   local indim = torch.random(1,4)
   local ini = torch.random(1,5)
   local inj = torch.random(1,5)
   local input = torch.zeros(nframes, indim, ini, inj):uniform()
   local module = nn.SpatialBatchNormalization(indim)

   local err = jac.testJacobian(module, input, -2, 4)
   mytester:assertlt(err,precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input,
                                      module.weight, module.gradWeight, -2, 4)
   mytester:assertlt(err,precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input,
                                      module.bias, module.gradBias, -2, 4)
   mytester:assertlt(err,precision, 'error on weight ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight, -2, 4)
   mytester:assertlt(err,precision, 'error on weight [direct update] ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias, -2, 4)
   mytester:assertlt(err,precision, 'error on bias [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input,
                                        'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                           'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input,
                                        'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                           'error on bias [%s]', t))
   end

   -- IO
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

   -- batch norm without affine transform
   module = nn.SpatialBatchNormalization(indim, 1e-5, 0.1, false)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   -- IO
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.GradientReversal()
   local ini = math.random(3,5)
   local inj = math.random(3,5)
   local ink = math.random(3,5)
   local input = torch.Tensor(ini,inj,ink):zero()
   -- Two GradientReversal layers should cancel each other out
   local module = nn.Sequential()
   module:add(nn.GradientReversal())
   module:add(nn.GradientReversal())

   local err = jac.testJacobian(module,input, 0.1, 10)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input, 0.1, 10)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Padding()
   local fanin = math.random(1,3)
   local sizex = math.random(4,16)
   local sizey = math.random(4,16)
   local pad = math.random(-3,3)
   local val = torch.randn(1):squeeze()
   local module = nn.Padding(1, pad, 3, val)
   local input = torch.rand(fanin,sizey,sizex)
   local size = input:size():totable()
   size[1] = size[1] + math.abs(pad)

   local output = module:forward(input)
   mytester:assertTableEq(size, output:size():totable(), 0.00001, "Padding size error")

   local gradInput = module:backward(input, output)
   mytester:assertTensorEq(gradInput, input, 0.00001, "Padding backward error")
end

function nntest.addSingletonDimension()
   local dims = torch.random(5)
   local size = torch.LongTensor(dims):random(10)
   local perm = torch.randperm(dims):totable()
   local tensor = torch.Tensor(table.unpack(size:totable())):uniform():permute(table.unpack(perm))
   size = torch.gather(size, 1, torch.LongTensor(perm))

   local firstDim = nn.utils.addSingletonDimension(tensor)
   mytester:assertTableEq(firstDim:size():totable(), {1, table.unpack(size:totable())},
                          "wrong size for singleton dimension 1")
   mytester:assertTensorEq(firstDim[1], tensor, 0,
                           "wrong content for singleton dimension 1")

   local dim = torch.random(dims + 1)
   local result = nn.utils.addSingletonDimension(tensor, dim)
   local resultSize = size:totable()
   table.insert(resultSize, dim, 1)
   mytester:assertTableEq(result:size():totable(), resultSize,
                          "wrong size for random singleton dimension")
   mytester:assertTensorEq(result:select(dim, 1), tensor, 0,
                           "wrong content for random singleton dimension")

   mytester:assertError(function() nn.utils.addSingletonDimension(tensor, dims + 2) end,
                        "invalid dimension not detected")
end

function nntest.Typecast()
  local function make_network()
    local seq = nn.Sequential()
    seq:add(nn.Linear(15, 10))
    seq:add(nn.Linear(15, 10))
    seq.modules[1].bias:fill(1)
    seq.modules[2].bias:fill(2)
    return seq
  end

  -- make sure that the typecasts aren't nops
  assert(torch.getdefaulttensortype() == 'torch.DoubleTensor')

  -- basic net
  local net = make_network()
  net.modules[1].empty_tensor = torch.Tensor()
  net:float()
  assert(net.modules[1].bias:type() == 'torch.FloatTensor',
      net.modules[1].bias:type())
  assert(net.modules[1].empty_tensor:type() == 'torch.FloatTensor')
  assert(net.modules[1].bias ~= net.modules[2].bias)
  net.modules[1].bias:fill(3)
  assert(net.modules[1].bias[1] == 3)
  assert(net.modules[2].bias[1] == 2)

  -- shared tensors remain shared
  local net = make_network()
  net.modules[2].bias = net.modules[1].bias
  net:float()
  assert(net.modules[1].bias:type() == 'torch.FloatTensor')
  assert(net.modules[1].bias == net.modules[2].bias)
  assert(net.modules[1].bias[1] == 1)

  -- shared storages remain shared
  local net = make_network()
  net.modules[2].bias:set(net.modules[1].bias)
  local net = net:float()
  assert(net.modules[1].bias:type() == 'torch.FloatTensor')
  assert(net.modules[1].bias ~= net.modules[2].bias)
  net.modules[1].bias:fill(3)
  assert(net.modules[1].bias[1] == 3)
  assert(net.modules[2].bias[1] == 3)

  -- tricky: overlapping views on the same storage are preserved
  local net = make_network()
  local overlap_storage = torch.Tensor(15):fill(1)
  net.modules[1].bias = overlap_storage:narrow(1, 1, 10)
  net.modules[2].bias = overlap_storage:narrow(1, 6, 10)
  net:float()
  assert(net.modules[1].bias:type() == 'torch.FloatTensor')
  assert(net.modules[1].bias ~= net.modules[2].bias)
  net.modules[1].bias:fill(3)
  assert(net.modules[1].bias[1] == 3)
  assert(net.modules[2].bias[1] == 3)
  assert(net.modules[2].bias[6] == 1) -- only the first 5 elements overlapped

  -- check recursiveType on a table
  local net1 = make_network()
  local net2 = make_network()
  net2.modules[1].bias:set(net1.modules[1].bias)
  net1:float()
  net2:float()
  net1.modules[1].bias:fill(3)
  assert(net2.modules[1].bias[1] == 1)

  local net1 = make_network()
  local net2 = make_network()
  net2.modules[1].bias:set(net1.modules[1].bias)

  local tensorCache = {}
  net1:type('torch.FloatTensor', tensorCache)
  net2:type('torch.FloatTensor', tensorCache)
  net1.modules[1].bias:fill(3)
  assert(net2.modules[1].bias[1] == 3)

  local net1 = make_network()
  local net2 = make_network()
  net2.modules[1].bias:set(net1.modules[1].bias)

  nn.utils.recursiveType({net1, net2}, 'torch.FloatTensor')
  net1.modules[1].bias:fill(3)
  assert(net2.modules[1].bias[1] == 3)

  -- smoke test some modules with custom type methods
  local custom_type_modules = {
    nn.MixtureTable(3),
    nn.ConcatTable(),
    nn.Copy(),
    nn.Copy(nil, nil, nil, true),
    nn.SpatialContrastiveNormalization(),
    nn.DotProduct(),
    nn.PairwiseDistance(1),
    nn.SpatialDivisiveNormalization(),
    nn.SpatialSubtractiveNormalization()
  }
  for _, module in ipairs(custom_type_modules) do
    module:float()
  end
end

function nntest.Module_apply()
  local s = nn.Sequential()
  s:add(nn.Linear(10,10))
  local s2 = nn.Sequential()
  s2:add(nn.Linear(10,5))
  s:add(s2)
  s:add(nn.Tanh())

  local seen = 0
  s:apply(function(module)
    if torch.type(module) == 'nn.Linear' then
      module.bias:resize(20)
      seen = seen + 1
    end
  end)
  mytester:asserteq(seen, 2)
  mytester:asserteq(s.modules[1].bias:size(1), 20)
  mytester:asserteq(s2.modules[1].bias:size(1), 20)
end

function nntest.Cosine()
   local inputSize = 4
   local outputSize = 5

   -- test 1D
   local input = torch.randn(inputSize)
   local gradOutput = torch.randn(outputSize)
   local cosine = nn.Cosine(inputSize,outputSize)
   local output = cosine:forward(input)
   local inputNorm = input:norm()+1e-12
   local weight2 = cosine.weight[2]
   local output2 = torch.dot(weight2, input)/((weight2:norm()+1e-12)*inputNorm)
   mytester:assert(math.abs(output2 - output[2]) < 0.000001,"Cosine output 1D err weight[2]")
   local output2 = torch.mv(cosine.weight, input)
   output2:cdiv(cosine.weight:norm(2,2)+1e-12):div(inputNorm)
   mytester:assertTensorEq(output, output2, 0.000001, "Cosine output 1D err")
   local gradInput = cosine:updateGradInput(input, gradOutput)
   local gradInput2 = gradInput:clone():zero()
   for j=1,outputSize do
      local w_j = cosine.weight[j]
      local nw_j = w_j:norm()+1e-12
      for i=1,inputSize do
         local w_ij = w_j[i]
         local grad_i = (w_ij/(inputNorm*nw_j))
         grad_i = grad_i - (output[j]*input[i]/(inputNorm*inputNorm))
         grad_i = grad_i * gradOutput[j]
         gradInput2[i] = gradInput2[i] + grad_i
      end
   end
   mytester:assertTensorEq(gradInput2, gradInput, 0.000001, "Cosine gradInput 1D err")
   cosine:zeroGradParameters()
   cosine:accGradParameters(input, gradOutput, 1)
   local gradWeight2 = cosine.weight:clone():zero()
   for j=1,outputSize do
      local w_j = cosine.weight[j]
      local nw_j = w_j:norm()+1e-12
      for i=1,inputSize do
         local w_ij = w_j[i]
         local gW_ij = (gradOutput[j]/nw_j)  * ( ( input[i] / inputNorm ) - (output[j] * w_ij / nw_j) )
         gradWeight2[{j,i}] = gW_ij
      end
   end
   mytester:assertTensorEq(cosine.gradWeight, gradWeight2, 0.000001, "Cosine gradWeight 2D err")

   -- test 2D
   local batchSize = 3
   local input = torch.randn(batchSize, inputSize)
   local gradOutput = torch.randn(batchSize, outputSize)
   cosine:zeroGradParameters()
   local cosine2 = cosine:clone()
   local output = cosine:forward(input)
   local output2 = cosine2:forward(input[2])
   mytester:assertTensorEq(output[2], output2, 0.000001, "Cosine output 2D err")
   local gradInput = cosine:backward(input, gradOutput)

   local gradInput2 = gradInput:clone():zero()
   for i=1,batchSize do
      cosine2:forward(input[i], gradOutput[i])
      gradInput2[i]:copy(cosine2:backward(input[i], gradOutput[i]))
   end
   mytester:assertTensorEq(gradInput, gradInput2, 0.000001, "Cosine gradInput 2D err")
   mytester:assertTensorEq(cosine.gradWeight, cosine2.gradWeight, 0.000001, "Cosine gradWeight 2D err")
end

mytester:add(nntest)

if not nn then
   require 'nn'
   jac = nn.Jacobian
   sjac = nn.SparseJacobian
   mytester:run()
else
   jac = nn.Jacobian
   sjac = nn.SparseJacobian
   function nn.test(tests)
      -- randomize stuff
       local seed = os.time()
       print('Seed: ', seed)
       math.randomseed(seed)
       torch.manualSeed(seed)
      mytester:run(tests)
      return mytester
   end
end
