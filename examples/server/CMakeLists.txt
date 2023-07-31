# server
set(TARGET server)
add_executable(${TARGET} server.cpp json.hpp httplib.h)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE common whisper ${CMAKE_THREAD_LIBS_INIT})

