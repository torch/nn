SET(src init.c)

FILE(GLOB luasrc *.lua)
SET(luasrc ${luasrc} test/test.lua)

ADD_TORCH_PACKAGE(nn "${src}" "${luasrc}" "Machine Learning")
ADD_TORCH_DOK(dok nn "Machine Learning" "Neural Networks" 3.1)

TARGET_LINK_LIBRARIES(nn luaT TH)
