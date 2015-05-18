nn.utils = {}

function nn.utils.recursiveType(param, type_str)
   if torch.type(param) == 'table' then
      for k, v in pairs(param) do
         param[k] = nn.utils.recursiveType(v, type_str)
      end
   elseif torch.isTypeOf(param, 'nn.Module') or
          torch.isTypeOf(param, 'nn.Criterion') then
      param:type(type_str)
   elseif torch.isTensor(param) then
       param = param:type(type_str)
   end
   return param
end

table.unpack = table.unpack or unpack
