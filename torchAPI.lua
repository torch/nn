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
  |  library | Name of overarching library |
  |  pattern | Regex for function handles  |
  |    tag   | API-FFI method identifier   |
  |   rtype  | Return type for FFI methods |
  |  prefix  | Function signature prefix   |
  ------------------------------------------

Default Format:
  <tag> <rtype> <library>_<pattern>(...)
  |_______        ________|
           prefix

By way of example, using the default 
pattern := '([%a%d_]+)':
  
  <tag>    <rtype>   <library>_<pattern>
  TH_API     void  THNN_Abs_updateOutput

Equivalently, if we initialize an API with
pattern := '%(([%a%d_]+)%)':
  
  <tag>    <rtype>    <library>_<pattern>
  TH_API    void    THNN_(Abs_updateOutput)


Authored: 2016-01-05 (jwilson)
Modified: 2016-02-04
--]]

---------------- External Dependencies
local ffi = require 'ffi'

------------------------------------------------
--                                           API
------------------------------------------------
local API = torch.class('torchAPI')

function API:__init(config, header)
  local L = config or {}
  assert(type(L.library) == 'string')
  L.pattern = L.pattern or '([%a%d_]+)'
  L.rtype   = L.rtype   or 'void'
  L.tag     = L.tag     or ''
  L.prefix  = L.prefix  or string.format('%s %s %s_', L.tag, L.rtype, L.library):gsub("^%s+", "")
  self.config = L 

  if header then
    self:c_init(header)
  end
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

function API:extract_handles(forward, pattern, handles)
  local handles = handles or {}
  local pattern = pattern or self.config.prefix ..self.config.pattern
  for handle in string.gmatch(forward, pattern) do
    handles[#handles+1] = handle
  end
  return handles
end

function API:c_init(header)

  ---- Define C Preamble
  if header['preamble'] then
    ffi.cdef(header['preamble'])
  end

  ---- Check for forward declaration
  forward = header['forward']
  if not forward then return end -- terminate if no forward

  ---- Swap in template Prepare forward for declaration / expansion of macros 
  local templates = header['template']
  if templates then
    if type(templates) == 'string' then
      templates = {templates}
    end

    for idx = 1, #templates do
      template = templates[idx]

      ---- Check for special patterns
      if template:match('<tag>') then
        template = template:gsub('<tag>', self.config.tag)
      end

      if template:match('<rtype>') then
        template = template:gsub('<rtype>', self.config.rtype)
      end

      if template:match('<library>') then
        template = template:gsub('<library>', self.config.library)
      end

      if template:match('$([0-9]+)') then
        template = template:gsub('$([0-9]+)', '%%%1')
      end

      forward = forward:gsub(self.config.prefix..self.config.pattern, template)
    end
  end

  ---- Expand macros
  local macros = header['macro'] or header['macros']
  if macros then
    local macros = self:_interp_macros(macros)
    local flag   = false

    -- Apply macros with same interpretation everywhere
    for old, new in pairs(macros) do
      if type(new) == 'string' then
        forward = forward:gsub(old, new)
      else
        flag = true -- Are there additional macros to process?
      end
    end

    -- Apply macros with alternative interpretations
    if flag then
      for id, group in pairs(macros) do
        if type(group) == 'table' then
          local forward = forward
          for old, new in pairs(group) do
            forward = forward:gsub(old, new)
          end
          ffi.cdef(forward)
        end
      end
    else
      -- No alternatives case
      ffi.cdef(forward)
    end
  else
    -- No macros case
    ffi.cdef(forward)
  end
end

function API:_interp_macros(macros, interp)
  local interp = interp or {}
  for key, macro in pairs(macros) do
    if type(macro) == 'string' then     -- Macros without same 
      assert(type(key) == 'string')     -- interpretation everywhere
      interp[key] = macro               -- must have string keys    
    else
      assert(type(macro) == 'table')    -- Non-global macros must be tables
      if type(key) == 'number' then     -- Pre-sorted group, i.e. all terms
        interp[key] = macros            -- in interpetation are already together 
      else
        assert(type(key) == 'string')   -- Single term with multiple interps
        for k, v in pairs(macro) do   
          assert(type(k) == 'string')   -- To avoid confusion with pre-sorted
          interp[k] = interp[k] or {}   -- groups; strings must be used to denote
          interp[k][key] = v            -- membership.
        end
      end
    end
  end
  return interp
end

function API:bind(cstruct, forward, ttype, getter, library, pattern, handles, prefix)
  local library = library or {}
  local handles = self:extract_handles(forward, pattern, handles)
  local prefix  = prefix or string.format('%s_%s', self.config.library, ttype)
  local getter  = getter or self.getState

  if type(cstruct) == 'string' then
    ---- Polyfill for LUA 5.1
    if not package.searchpath then
      local sep = package.config:sub(1,1)
      function package.searchpath(mod, path)
        local mod, nm, f = mod:gsub('%.', sep)
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
    if cstruct[prefix .. h] then  -- implicitly add state
      library[h] = function(...) cstruct[prefix .. h](getter(), ...) end
    else
      print('> Warning:::Unable to locate method ' .. prefix .. h)
    end
  end

  return library
end
