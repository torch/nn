require 'torch'

local mytester = torch.Tester()
local jac

local precision = 1e-5
local expprecision = 1e-4

local nntest = {}
local nntestx = {}

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
   local ini = math.random(50,70)
   local inj = math.random(50,70)
   local input = torch.Tensor(ini):zero()
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
   mytester:assertlt(err,expprecision, 'error on state ')

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
   
   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
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

function nntestx.TemporalMaxPooling()
   local from = math.random(1,10)
   local ki = math.random(1,10)
   local si = math.random(1,4)
   local outi = math.random(10,20)
   local ini = (outi-1)*si+ki
   local module = nn.TemporalMaxPooling(ki, si)
   local input = torch.Tensor(ini, from):zero()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
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

mytester:add(nntest)

if not nn then
   require 'nn'
   jac = nn.Jacobian
   mytester:run()
else
   jac = nn.Jacobian
   function nn.test(tests)
      -- randomize stuff
      math.randomseed(os.time())
      mytester:run(tests)
   end
end
