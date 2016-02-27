local Identity, _ = torch.class('nn.Identity', 'nn.Module')

local function isSameType(a, b)
   return torch.type(a) == torch.type(b)
end

local function identity(out, input)
   if torch.isTensor(input) then
      -- out could have been a table in a previous call
      if not isSameType(out, input) then
         out = input.new()
      end
      out:set(input)
   elseif torch.type(input) ~= 'table' then
      -- copy numbers, strings or functions
      out = input
   else
      -- avoid recreating tables and tensors
      -- every time the function is called
      if not isSameType(out, input) then
         out = {}
      end
      -- recursively sets the entries of out
      for k,v in pairs(input) do
         out[k] = identity(out[k], v)
      end
      -- clean up behind if the current input is
      -- smaller than the previous one
      for k in pairs(out) do
         if not input[k] then
            out[k] = nil
         end
      end
   end
   return out
end

function Identity:updateOutput(input)
   self.output = identity(self.output, input)
   return self.output
end


function Identity:updateGradInput(input, gradOutput)
   self.gradInput = identity(self.gradInput, gradOutput)
   return self.gradInput
end
