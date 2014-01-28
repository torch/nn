CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)

INCLUDE(CheckCCompilerFlag)
CHECK_C_COMPILER_FLAG(-pedantic C_HAS_PEDANTIC)
IF (C_HAS_PEDANTIC)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pedantic")
ENDIF ()

SET(src init.c)

FILE(GLOB luasrc *.lua)
SET(luasrc ${luasrc} test/test.lua)

ADD_TORCH_PACKAGE(nn "${src}" "${luasrc}")
ADD_TORCH_DOK(dok nn "Machine Learning" "Neural Networks" 3.1)

TARGET_LINK_LIBRARIES(nn luaT TH)
