local Identity, _ = torch.class('nn.Identity', 'nn.Module')

function Identity:updateOutput(input)
   self.output = input
   return self.output
end


function Identity:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end

function Identity:clearState()
   -- don't call set because it might reset referenced tensors
   local function clear(t)
      if torch.isTensor(t) then
         return t.new()
      elseif type(t) == 'table' then
         local cleared = {}
         for k,v in pairs(t) do
            cleared[k] = clear(v)
         end
         return cleared
      else
         return nil
      end
   end
   if self.output then self.output = clear(self.output) end
   if self.gradInput then self.gradInput = clear(self.gradInput) end
   return self
end
