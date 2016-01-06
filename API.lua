------------------------------------------------
--                                      Preamble
------------------------------------------------
--[[
Generic application program interface (API) for
convenient linkage of C and Lua code libraries.

Glossary:
  ------------------------------------------
  | Term     | Description                 |
  ------------------------------------------
  |  libname | Name of overarching library |
  |  pattern | Regex for function handles  |
  |    tag   | API-FFI method identifier   |
  |   rtype  | Return type for FFI methods |
  |  prefix  | Function signature prefix   |
  ------------------------------------------

Default Format:
  <tag> <rtype> <libname>_<pattern>(...)
  |_______        ________|
           prefix

By way of example, using the default 
pattern := '([%a%d_]+)':
  
  <tag>    <rtype>   <libname>_<pattern>
  TH_API     void  THNN_Abs_updateOutput

Equivalently, if we initialize an API with
pattern := '%(([%a%d_]+)%)':
  
  <tag>    <rtype>    <libname>_<pattern>
  TH_API     void  THNN_(Abs_updateOutput)



Authored: 2016-01-05 (jwilson)
Modified: 2016-01-06
--]]

---------------- External Dependencies
local ffi  = require 'ffi'

------------------------------------------------
--                                           API
------------------------------------------------
local API = torch.class('API')

function API:__init(config)
  local L = config or {}
  assert(type(L.libname) == 'string')
  L.pattern = L.pattern or '([%a%d_]+)'
  L.rtype   = L.rtype   or 'void'
  L.tag     = L.tag     or '' -- recommended: 'TH_API'
  L.prefix  = L.prefix  or string.format('%s %s %s_', L.tag, L.rtype, L.libname):gsub("^%s+", "")
  self.config = L 
end

function API:getState(ttype)
  if ttype == 'Cuda' then
    if not self.cuda_ptr then 
      self.cuda_ptr = ffi.typeof('THCState*')
    end
    return self.cuda_ptr(cutorch.getState())
  else
    return ffi.NULL or nil
  end
end

function API:extract_handles(header, pattern, handles)
  local handles = handles or {}
  local pattern = pattern or self.config.prefix .. self.config.pattern
  for handle in string.gmatch(header, pattern) do
    handles[#handles+1] = handle
  end
  return handles
end

function API:forward_declare(header, macros, prep)
  local header = header

  ---- Prepare header for declaration / expansion of macros 
  if prep then
    if type(prep) == 'table' then
      for idx = 1, #prep/2 do
        header = header:gsub(prep[idx], prep[idx+1])
      end
    elseif type(prep) == 'string' then
      header = header:gsub(self.config.prefix..self.config.pattern, prep)
    else
      header = header:gsub(self.config.tag, '') -- just strip off the tag
    end
  end

  ---- Expand macros (if provided)
  if macros then
    for idx, macro in pairs(macros) do
      local header = header
      for old, new in pairs(macro) do
        header = header:gsub(old, new)
      end
      ffi.cdef(header)
    end
  else
    ffi.cdef(header)
  end
end

function API:bind(cstruct, header, ttype, getter, library, pattern, handles, prefix)
  local library = library or {}
  local handles = self:extract_handles(header, pattern, handles)
  local prefix  = prefix or string.format('%s_%s', self.config.libname, ttype)
  local getter  = getter or self.getState

  if type(cstruct) == 'string' then
    ---- Polyfill for LUA 5.1
    if not package.searchpath then
      local sep = package.config:sub(1,1)
      function package.searchpath(mod, path)
        local mod ,nm, f = mod:gsub('%.', sep)
        for m in path:gmatch('[^;]+') do
          nm = m:gsub('?', mod)
          f  = io.open(nm, 'r')
          if f then f:close() return nm end
        end
      end
    end
    cstruct = ffi.load(package.searchpath(cstruct, package.cpath))
  end

  for idx, h in ipairs(handles) do
    if cstruct[prefix .. h] then  -- implicitely add state
      library[h] = function(...) cstruct[prefix .. h](getter(), ...) end
    else
      print('> Warning:::Unable to locate method ' .. prefix .. h)
    end
  end

  return library
end
