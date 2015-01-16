-- This is code common to container modules, which are collections of
-- smaller constituent modules like Parallel, Sequential, etc.
local Container, parent = torch.class('nn.Container', 'nn.Module')

function Container:__init(...)
    parent.__init(self, ...)
    self.modules = {}
end

function Container:add(module)
    table.insert(self.modules, module)
    return self
end

function Container:get(index)
    return self.modules[index]
end

function Container:size()
    return #self.modules
end

function Container:zeroGradParameters()
    for i=1,#self.modules do
        self.modules[i]:zeroGradParameters()
    end
end

function Container:updateParameters(learningRate)
    for _,module in ipairs(self.modules) do
        module:updateParameters(learningRate)
    end
end

function Container:training()
    for i=1,#self.modules do
        self.modules[i]:training()
    end
end

function Container:evaluate()
    for i=1,#self.modules do
        self.modules[i]:evaluate()
    end
end

function Container:share(mlp, ...)
    for i=1,#self.modules do
        self.modules[i]:share(mlp.modules[i], ...);
    end
end

function Container:reset(stdv)
    for i=1,#self.modules do
        self.modules[i]:reset(stdv)
    end
end

function Container:parameters()
    local function tinsert(to, from)
        if type(from) == 'table' then
            for i=1,#from do
                tinsert(to,from[i])
            end
        else
            table.insert(to,from)
        end
    end
    local w = {}
    local gw = {}
    for i=1,#self.modules do
        local mw,mgw = self.modules[i]:parameters()
        if mw then
            tinsert(w,mw)
            tinsert(gw,mgw)
        end
    end
    return w,gw
end
