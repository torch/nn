local Index, parent = torch.class('nn.Index', 'nn.Module')

function Index:__init(dimension)
    parent.__init(self)
    self.dimension = dimension
    self.gradInput = {self.gradInput}
end

function Index:updateOutput(input)
    local t = input[1]
    local index = input[2]
    self.output:index(t, self.dimension, index)
    return self.output
end

function Index:updateGradInput(input, gradOutput)
    local t = input[1]
    local index = input[2]

    local gradInput = self.gradInput[1] -- no gradient for the index variable
    gradInput:resizeAs(t):zero()
    for n = 1, index:size(1) do
        gradInput:narrow(self.dimension, index[n], 1):add(gradOutput:narrow(self.dimension, n, 1))
    end
    return self.gradInput
end

