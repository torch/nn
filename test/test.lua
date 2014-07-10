require 'torch'

local mytester = torch.Tester()
local jac
local sjac

local precision = 1e-5
local expprecision = 1e-4

local nntest = {}

function nntest.Add()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.Add(ini*inj*ink)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')
   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err,precision, 'error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err,precision, 'error on bias [direct update]')

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'error on bias [%s]', t))
   end

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.CMul()
   local ini = math.random(5,15)
   local inj = math.random(5,15)
   local ink = math.random(5,15)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.CMul(ini*inj*ink)

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

function nntest.Dropout()
   local p = 0.2 --prob of droping out a neuron
   local input = torch.Tensor(1000):fill((1-p))
   local module = nn.Dropout(p)
   -- version 2
   local output = module:forward(input)
   mytester:assert(math.abs(output:mean() - (1-p)) < 0.05, 'dropout output')
   local gradInput = module:backward(input, input)
   mytester:assert(math.abs(gradInput:mean() - (1-p)) < 0.05, 'dropout gradInput')
   -- version 1 (old nnx version)
   local input = input:fill(1)
   local module = nn.Dropout(p,true)
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
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.Exp()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Log()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.Log()

   local err = jac.testJacobian(module,input, 0.1, 10)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input, 0.1, 10)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.HardTanh()
   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()
   
   local module = nn.HardTanh()
   
   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision ,  'error on state ')
   
   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Abs()
   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()
   
   local module = nn.Abs()
   
   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision ,  'error on state ')
   
   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Threshold()
   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.Threshold(torch.uniform(-2,2),torch.uniform(-2,2))

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.HardShrink()
   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.HardShrink(math.random()/2)

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SoftShrink()
   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.SoftShrink(math.random()/2)

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Power()
   local in1 = torch.rand(10,20)
   local module = nn.Power(2)
   local out = module:forward(in1)
   local err = out:dist(in1:cmul(in1))
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local pw = torch.uniform()*math.random(1,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.Power(pw)

   local err = nn.Jacobian.testJacobian(module, input, 0.1, 2)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module,input, 0.1, 2)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Square()
   local in1 = torch.rand(10,20)
   local module = nn.Square()
   local out = module:forward(in1)
   local err = out:dist(in1:cmul(in1))
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.Square()

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Sqrt()
   local in1 = torch.rand(10,20)
   local module = nn.Sqrt()
   local out = module:forward(in1)
   local err = out:dist(in1:sqrt())
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.Sqrt()

   local err = nn.Jacobian.testJacobian(module, input, 0.1, 2)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input, 0, 2)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Linear()
   local ini = math.random(5,7)
   local inj_vals = {math.random(5,7), 1}  -- Also test the inj = 1 spatial case
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
   local ini = math.random(5000,10000)
   local inj = math.random(50,100)
   local numNonzero = math.random(5,20)
   
   local module = nn.SparseLinear(ini,inj)

   -- Create a random sparse vector
   N = {}
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

function nntest.Euclidean()
   local ini = math.random(50,70)
   local inj = math.random(50,70)
   local input = torch.Tensor(ini):zero()
   local module = nn.Euclidean(ini,inj)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.WeightedEuclidean()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local input = torch.Tensor(ini):zero()
   local module = nn.WeightedEuclidean(ini,inj)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err,precision, 'error on bias ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

local function criterionJacobianTest1D(cri, input, target)
   local eps = 1e-6
   local fx = cri:forward(input, target)
   local dfdx = cri:backward(input, target)
   -- for each input perturbation, do central difference
   local centraldiff_dfdx = torch.Tensor():resizeAs(dfdx)
   for i=1,input:size(1) do
      -- f(xi + h)
      input[i] = input[i] + eps
      local fx1 = cri:forward(input, target)
      -- f(xi - h)
      input[i] = input[i] - 2*eps
      local fx2 = cri:forward(input, target)   
      -- f'(xi) = (f(xi + h) - f(xi - h)) / 2h
      local cdfx = (fx1 - fx2) / (2*eps)
      -- store f' in appropriate place
      centraldiff_dfdx[i] = cdfx
      -- reset input[i]
      input[i] = input[i] + eps
   end

   -- compare centraldiff_dfdx with :backward()
   local err = (centraldiff_dfdx - dfdx):abs():max()
   mytester:assertlt(err, precision, 'error in difference between central difference and :backward')
end

function nntest.MSECriterion()
   local input = torch.rand(100)
   local target = input:clone():add(torch.rand(100))
   local cri = nn.MSECriterion()
   criterionJacobianTest1D(cri, input, target)   
end

function nntest.WeightedMSECriterion()
   local input = torch.rand(100)
   local target = input:clone():add(torch.rand(100))
   local cri = nn.WeightedMSECriterion(torch.rand(100))
   criterionJacobianTest1D(cri, input, target)
end

function nntest.BCECriterion()
   local input = torch.rand(100)
   local target = input:clone():add(torch.rand(100))
   local cri = nn.BCECriterion()
   criterionJacobianTest1D(cri, input, target)
end

function nntest.LogSigmoid()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.LogSigmoid()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.LogSoftmax()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local input = torch.Tensor(ini,inj):zero()
   local module = nn.LogSoftMax()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,1e-3, 'error on state ')

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
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj*ink):zero()
   local module = nn.Max(1)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Min()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj*ink):zero()
   local module = nn.Min(1)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Mean()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.Mean(torch.random(1,3))

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Mul()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.Mul(ini*inj*ink)

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
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.Sigmoid()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Softmax()
   local ini = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ink, ini):zero()
   local module = nn.SoftMax()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,expprecision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Softmin()
   local ini = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ink, ini):zero()
   local module = nn.SoftMin()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,expprecision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Softsign()
   local ini = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ink, ini):zero()
   local module = nn.SoftSign()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SoftPlus()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.SoftPlus()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SpatialSubtractiveNormalization_2dkernel()
   local inputSize = math.random(11,20)
   local kersize = 9
   local nbfeatures = math.random(5,10)
   local kernel = torch.Tensor(kersize,kersize):fill(1)
   local module = nn.SpatialSubtractiveNormalization(nbfeatures,kernel)
   local input = torch.rand(nbfeatures,inputSize,inputSize)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SpatialSubtractiveNormalization_1dkernel()
   local inputSize = math.random(11,20)
   local kersize = 9
   local nbfeatures = math.random(5,10)
   local kernel = torch.Tensor(kersize):fill(1)
   local module = nn.SpatialSubtractiveNormalization(nbfeatures,kernel)
   local input = torch.rand(nbfeatures,inputSize,inputSize)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SpatialDivisiveNormalization_2dkernel()
   local inputSize = math.random(11,20)
   local kersize = 9
   local nbfeatures = math.random(5,10)
   local kernel = torch.Tensor(kersize,kersize):fill(1)
   local module = nn.SpatialDivisiveNormalization(nbfeatures,kernel)
   local input = torch.rand(nbfeatures,inputSize,inputSize)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SpatialDivisiveNormalization_1dkernel()
   local inputSize = math.random(11,20)
   local kersize = 9
   local nbfeatures = math.random(5,10)
   local kernel = torch.Tensor(kersize):fill(1)
   local module = nn.SpatialDivisiveNormalization(nbfeatures,kernel)
   local input = torch.rand(nbfeatures,inputSize,inputSize)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SpatialConvolution()
   local from = math.random(1,10)
   local to = math.random(1,10)
   local ki = math.random(1,10)
   local kj = math.random(1,10)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(10,20)
   local outj = math.random(10,20)
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
   local from = math.random(1,10)
   local to = math.random(1,10)
   local ki = math.random(1,5)
   local kj = math.random(1,5)
   local outi = math.random(10,20)
   local outj = math.random(10,20)
   local ini = outi-1+ki
   local inj = outj-1+kj
   local module = nn.SpatialConvolutionMM(from, to, ki, kj)
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
   outi = math.random(4,8)
   outj = math.random(4,8)
   ini = outi-1+ki
   inj = outj-1+kj
   module = nn.SpatialConvolutionMM(from, to, ki, kj)
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
end

function nntest.SpatialConvolutionMap()
   local from = math.random(1,10)
   local fanin = math.random(1, from)
   local to = math.random(1,10)
   local ki = math.random(1,10)
   local kj = math.random(1,10)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(10,20)
   local outj = math.random(10,20)
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


function nntest.SpatialFullConvolution()
   local from = math.random(1,10)
   local to = math.random(1,10)
   local ki = math.random(1,10)
   local kj = math.random(1,10)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local ini = math.random(10,18)
   local inj = math.random(10,18)
   local module = nn.SpatialFullConvolution(from, to, ki, kj, si, sj)
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
   local batch = math.random(2,5)
   ini = math.random(4,8)
   inj = math.random(4,8)
   module = nn.SpatialFullConvolution(from, to, ki, kj, si, sj)
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
end

function nntest.SpatialFullConvolutionMap()
   local from = math.ceil(torch.uniform(2,5))
   local to = math.ceil(torch.uniform(2,7))
   local fanin = math.ceil(torch.uniform(1, from))
   local tt = nn.tables.random(from, to, fanin)
   local ki = math.ceil(torch.uniform(2,7))
   local kj = math.ceil(torch.uniform(2,7))
   local si = math.ceil(torch.uniform(1,3))
   local sj = math.ceil(torch.uniform(1,3))
   local ini = math.ceil(torch.uniform(10,18))
   local inj = math.ceil(torch.uniform(10,18))
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
    local from = math.ceil(torch.uniform(2,5))
    local to = math.ceil(torch.uniform(2,7))
    local tt = nn.tables.full(from, to)
    local ki = math.ceil(torch.uniform(2,7))
    local kj = math.ceil(torch.uniform(2,7))
    local si = math.ceil(torch.uniform(1,3))
    local sj = math.ceil(torch.uniform(1,3))
    local ini = math.ceil(torch.uniform(10,18))
    local inj = math.ceil(torch.uniform(10,18))
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
   local bs = torch.LongStorage(sin:size():size()+1)
   bs[1] = 1
   for i=1,sin:size():size() do bs[i+1] = sin:size()[i] end
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
   local from = math.random(1,10)
   local to = math.random(1,10)
   local ki = math.random(1,10)
   local kj = math.random(1,10)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(10,20)
   local outj = math.random(10,20)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local module = nn.SpatialConvolution(from, to, ki, kj, si, sj)
   local input = torch.randn(from,inj,ini)

   batchcompare(module,input, {'weight','bias','gradWeight','gradBias'})
end

function nntest.SpatialFullConvolutionBatchCompare()
   local from = math.random(1,10)
   local to = math.random(1,10)
   local ki = math.random(1,10)
   local kj = math.random(1,10)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local ini = math.random(10,18)
   local inj = math.random(10,18)

   local module = nn.SpatialFullConvolution(from, to, ki, kj, si, sj)
   local input = torch.randn(from, inj, ini)

   batchcompare(module,input, {'weight','bias','gradWeight','gradBias'})
end
   


function nntest.SpatialSubSamplingBatchCompare()
   local from = math.random(1,10)
   local ki = math.random(1,5)
   local kj = math.random(1,5)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(10,20)
   local outj = math.random(10,20)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local module = nn.SpatialSubSampling(from, ki, kj, si, sj)
   local input = torch.randn(from,inj,ini)--torch.Tensor(from, inj, ini):zero()

   batchcompare(module,input, {'weight','bias','gradWeight','gradBias'})
end

function nntest.SpatialSubSampling()
   local from = math.random(1,10)
   local ki = math.random(1,5)
   local kj = math.random(1,5)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(10,20)
   local outj = math.random(10,20)
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

   --verbose = true
   local batch = math.random(2,5)
   outi = math.random(4,8)
   outj = math.random(4,8)
   ini = (outi-1)*si+ki
   inj = (outj-1)*sj+kj
   module = nn.SpatialSubSampling(from, ki, kj, si, sj)
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
   local from = math.random(1,5)
   local ki = math.random(1,5)
   local kj = math.random(1,5)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(10,20)
   local outj = math.random(10,20)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local module = nn.SpatialMaxPooling(ki,kj,si,sj)
   local input = torch.rand(from,ini,inj)

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

   -- batch
   local nbatch = math.random(2,5)
   input = torch.rand(nbatch,from,ini,inj)
   module = nn.SpatialMaxPooling(ki,kj,si,sj)

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state (Batch) ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err (Batch) ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err (Batch) ')

end

function nntest.SpatialLPPooling()
   local fanin = math.random(1,4)
   local osizex = math.random(1,4)
   local osizey = math.random(1,4)
   local p = 2
   local mx = math.random(2,8)
   local my = math.random(2,8)
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
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.Sum(torch.random(1,3))

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Tanh()
   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
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
   local from = math.random(1,10)
   local to = math.random(1,10)
   local ki = math.random(1,10)
   local si = math.random(1,4)
   local outi = math.random(10,20)
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
   local nBatchFrame = 8
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
   local ki = math.random(1,10)
   local si = math.random(1,4)
   local outi = math.random(10,20)
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
   local from = math.random(10,10)
   local ki = math.random(5,10)
   local si = math.random(1,2)
   local outi = math.random(50,90)
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

function nntest.VolumetricConvolution()
   local from = math.random(2,5)
   local to = math.random(2,5)
   local kt = math.random(3,7)
   local ki = math.random(3,7)
   local kj = math.random(3,7)
   local st = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outt = math.random(3,7)
   local outi = math.random(3,7)
   local outj = math.random(3,7)
   local int = (outt-1)*st+kt
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local module = nn.VolumetricConvolution(from, to, kt, ki, kj, st, si, sj)
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

function nntest.VolumetricMaxPooling()
   local from = math.random(2,5)
   local to = from
   local kt = math.random(3,7)
   local ki = math.random(3,7)
   local kj = math.random(3,7)
   local st = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outt = math.random(3,7)
   local outi = math.random(3,7)
   local outj = math.random(3,7)
   local int = (outt-1)*st+kt
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local module = nn.VolumetricMaxPooling(kt, ki, kj, st, si, sj)
   local input = torch.Tensor(from, int, inj, ini):zero()
   
   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')
      
   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
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
   local p = n:getParameters()

   n:add( nn.Linear(10,10) )
   p = n:getParameters()

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
   local p = n:getParameters()

   n:add(nn.Linear(10,10))
   p = n:getParameters()

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
   local p = n:getParameters()

   n:add(nn.Linear(10,10))
   p = n:getParameters()

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
   local p = n:getParameters()

   n:add(nn.Linear(10,10))
   p = n:getParameters()

   local n1 = nn.Sequential()
   n1:add( nn.Linear(10,10) )

   local n2 = nn.Sequential()
   n2:add( nn.Linear(10,10) )

   local n = nn.Sequential()
   n:add( n1 )
   n:add( n2 )

   local p = n:getParameters()

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

function nntest.PairwiseDistance()
   -- Note: testJacobian doesn't support table inputs, and rather than re-write
   -- it so that it does, I'll just use a split table module on the input.
   -- I assume both SplitTable and Sequential do not have bugs, otherwise this
   -- test will break.
   for p = 1,4 do  -- test a few Lp norms
      -- TEST CASE 1: non-batch input, same code path but includes a resize
      local ini = math.random(10,20)
      local input = torch.Tensor(2, ini):zero()
      local module = nn.Sequential()
      module:add(nn.SplitTable(1))
      module:add(nn.PairwiseDistance(p))

      local err = jac.testJacobian(module,input)
      mytester:assertlt(err,precision, ' error on state ')
 
      local ferr,berr = jac.testIO(module,input)
      mytester:asserteq(ferr, 0, torch.typename(module)..' - i/o forward err ')
      mytester:asserteq(berr, 0, torch.typename(module)..' - i/o backward err ')

      -- Also check that the forward prop result is correct.
      input = torch.rand(2, ini)
      err = torch.dist(input:select(1,1), input:select(1,2), p) - 
        module:forward(input)[1]
      mytester:assertlt(err,precision, ' error on non-batch fprop ') 
 
      -- TEST CASE 2: batch input
      local inj = math.random(10,20)
      input = torch.Tensor(2, inj, ini):zero()

      -- (Rebuild the module to avoid correlated tests)
      module = nn.Sequential()
      module:add(nn.SplitTable(1))
      module:add(nn.PairwiseDistance(p))

      err = jac.testJacobian(module,input)
      mytester:assertlt(err,precision, ' error on state ')

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

function nntest.LookupTable()
   local totalIndex = math.random(10,100)
   local nIndex = math.random(5,7)
   local entry_size = math.random(5,7)
   local input = torch.IntTensor(nIndex):zero()
   local module = nn.LookupTable(totalIndex, entry_size)
   local minval = 1
   local maxval = totalIndex

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
   local nframe = math.random(50,70)
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
end
 
function nntest.AddConstant()
  local nbatch = torch.random(3, 5)
  local f = torch.random(3, 5)
  local h = torch.random(10,20)
  local w = torch.random(10,20)
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
end

function nntest.MulConstant()
  local nbatch = torch.random(3, 5)
  local f = torch.random(3, 5)
  local h = torch.random(10,20)
  local w = torch.random(10,20)
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
   local function equal(t1, t2, msg)
      if (torch.type(t1) == "table") then
         for k, v in pairs(t2) do
            equal(t1[k], t2[k])
         end
      else
         mytester:assertTensorEq(t1, t2, 0.00001, msg)
      end
   end
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
   module:float()
   local idx = #input
   local output = module:forward(input)
   equal(output, input[idx], "type output")
   local gradInput = module:backward(input, gradOutputs[idx])
   equal(gradInput[idx], gradOutputs[idx], "gradInput[idx] dimension " .. idx)
   equal(gradInput[nonIdx[idx]], zeros[nonIdx[idx]], "gradInput[nonIdx] dimension " .. idx)
end

function nntest.View()
   local input = torch.rand(10)
   local template = torch.rand(5,2)
   local target = template:size():totable()
   local module = nn.View(template:size())
   mytester:assertTableEq(module:forward(input):size():totable(), target, "Error in forward (1)")
   local module = nn.View(unpack(target))
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
    local input = torch.Tensor(unpack(shape)):zero()

    local err = jac.testJacobian(m, input)
    mytester:assertlt(err, precision, ' error on state ')

    local ferr, berr = jac.testIO(m, input)
    mytester:asserteq(ferr, 0, torch.typename(m)..' - i/o forward err ')
    mytester:asserteq(berr, 0, torch.typename(m)..' - i/o backward err ')
  end
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
      math.randomseed(os.time())
      mytester:run(tests)
      return mytester
   end
end
