local Square, parent = torch.class('nn.Square','nn.Module')

function Square:__init(args)
   parent.__init(self)
end

function Square:updateOutput(input)
   return input.nn.Square_updateOutput(self, input)
end

function Square:updateGradInput(input, gradOutput)
   return input.nn.Square_updateGradInput(self, input, gradOutput)
end
