local ZeroGrad, parent
if nn.ZeroGrad then -- prevent name conflicts with nnx
   ZeroGrad, parent = nn.ZeroGrad, nn.Module
else
   ZeroGrad, parent = torch.class('nn.ZeroGrad', 'nn.Module')
end

local function recursiveZero(t1,t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = recursiveZero(t1[key], t2[key])
      end
   elseif torch.isTensor(t2) then
      t1 = torch.isTensor(t1) and t1 or t2.new()
      t1:resizeAs(t2):zero()
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end

function ZeroGrad:updateOutput(input)
   self.output:set(input)
   return self.output
end

-- the gradient is simply zeroed.
-- useful when you don't want to backpropgate through certain paths.
function ZeroGrad:updateGradInput(input, gradOutput)
   self.gradInput = recursiveZero(self.gradInput, gradOutput)
   return self.gradInput
end
