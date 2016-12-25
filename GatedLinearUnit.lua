local GatedLinearUnit, parent = torch.class('nn.GatedLinearUnit', 'nn.Module')

function GatedLinearUnit:__init(dim)
   parent.__init(self)
   self.sigmoid = nn.Sigmoid()
   self.dim = dim
end

function GatedLinearUnit:updateOutput(input)
    local dim = self.dim or input:dim()
    local inputSize = input:size(dim)

    assert(inputSize % 2 == 0, "halving dimension needs to be even")

    self.fHalf = input:narrow(dim, 1, inputSize/2)
    self.sHalf = input:narrow(dim, inputSize/2 + 1, inputSize/2)

    self.sHalfOut = self.sigmoid:forward(self.sHalf)
    self.output:resizeAs(self.fHalf):copy(self.fHalf):cmul(self.sHalfOut)

    return self.output
end

function GatedLinearUnit:updateGradInput(input, gradOutput)
    local dim = self.dim or input:dim()
    local inputSize = input:size(dim)

    assert(inputSize % 2 == 0, "halving dimension needs to be even")

    local fGradInput = self.sHalfOut
    local sGradInput = self.sigmoid:backward(self.sHalf, gradOutput)
                                   :cmul(self.fHalf)

    self.gradInput:resizeAs(input)
    self.gradInput:narrow(dim, 1, inputSize/2)
                    :copy(fGradInput)
                    :cmul(gradOutput)
    self.gradInput:narrow(dim, inputSize/2+1, inputSize/2)
                    :copy(sGradInput)

    return self.gradInput
end
