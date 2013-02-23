local Copy, parent = torch.class('nn.Copy', 'nn.Module')

function Copy:__init(intype, outtype)
   intype = intype or torch.Tensor.__typename
   outtype = outtype or torch.Tensor.__typename

   parent.__init(self)
   self.gradInput = torch.getmetatable(intype).new()
   self.output = torch.getmetatable(outtype).new()

   if intype == outtype then

      self.updateOutput = function(self, input)
                        self.output = input
                        return input
                     end

      self.updateGradInput = function(self, input, gradOutput)
                         self.gradInput = gradOutput
                         return gradOutput
                      end
   end
end

function Copy:updateOutput(input)
   self.output:resize(input:size()):copy(input)
   return self.output
end

function Copy:updateGradInput(input, gradOutput)
   self.gradInput:resize(gradOutput:size()):copy(gradOutput)
   return self.gradInput
end
