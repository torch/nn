
local FeatureLPPooling, parent =
   torch.class('nn.FeatureLPPooling', 'nn.Module')

--[[
   Possible inputs that we handle:

   #### `batch_mode = false`
   The dimensionality of the input chooses between the following modes:

   ```
   [feature dim]
   [feature dim][opt dim 1]
   [feature dim][opt dim 1][opt dim 2]
   ```

   #### `batch_mode = true`
   The dimensionality of the input chooses between the following modes:
   ```
   [batch dim][feature dim]
   [batch dim][feature dim][opt dim 1]
   [batch dim][feature dim][opt dim 1][opt dim 2]
   ```

   The output has the same number of dimensions as the input, except the feature
   dimension size is reduced to ((`input` - `width`) / `stride`) + 1
]]
function FeatureLPPooling:__init(width, stride, power, batch_mode)
   parent.__init(self)

   if (width < 2 or width > 16) then
      error('width must be within 2 to 16')
   end

   if (stride < 1 or stride > 4) then
      error('stride must be within 1 to 4')
   end

   self.width = width
   self.stride = stride
   self.power = power
   self.batch_mode = batch_mode

   self.output = torch.Tensor()
   self.gradInput = torch.Tensor()
end

function FeatureLPPooling:updateOutput(input)
   input.THNN.FeatureLPPooling_updateOutput(input:cdata(),
                                            self.output:cdata(),
                                            self.power,
                                            self.width,
                                            self.stride,
                                            self.batch_mode)
   return self.output
end

function FeatureLPPooling:updateGradInput(input, gradOutput)
   input.THNN.FeatureLPPooling_updateGradInput(gradOutput:cdata(),
                                               input:cdata(),
                                               self.output:cdata(),
                                               self.gradInput:cdata(),
                                               self.power,
                                               self.width,
                                               self.stride,
                                               self.batch_mode)
   return self.gradInput
end

function FeatureLPPooling:__tostring__()
   return string.format('%s(w%d s%d power %d batch %d',
                        torch.type(self),
                        self.width, self.stride, self.power, self.batch_mode)
end
