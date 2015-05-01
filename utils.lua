nn._utils = {}

function nn._utils.recursiveType(param, type_str)
   if torch.type(param) == 'table' then
      for k, v in pairs(param) do
         param[k] = nn._utils.recursiveType(v, type_str)
      end
   elseif torch.isTensor(param) then
       param = param:type(type_str)
   end
   return param
end
