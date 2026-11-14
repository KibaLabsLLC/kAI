cmake_minimum_required(VERSION 3.18)
project(kibavoice LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Pinned upstream tags. Bump deliberately: llama.h changes between releases.
set(LLAMA_CPP_TAG   "v1.0.0" CACHE STRING "llama.cpp git tag")
set(WHISPER_CPP_TAG "v1.9.5" CACHE STRING "whisper.cpp git tag")

include(FetchContent)
include(ExternalProject)

# --- llama.cpp, linked as a static library ---------------------------------
set(BUILD_SHARED_LIBS      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP        OFF CACHE BOOL "" FORCE)
FetchContent_Declare(llama
  GIT_REPOSITORY https://github.com/ggml-org/llama.cpp
  GIT_TAG        ${LLAMA_CPP_TAG}
  GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(llama)

add_executable(kibavoice src/main.cpp)
target_link_libraries(kibavoice PRIVATE llama)

# --- whisper.cpp, built separately as the `whisper-cli` helper -------------
# whisper.cpp and llama.cpp each bundle their own ggml; linking both into one
# binary collides, so STT runs as a child process built in its own tree.
option(KIBAVOICE_BUILD_WHISPER "Build whisper-cli (speech-to-text)" ON)
if(KIBAVOICE_BUILD_WHISPER)
  ExternalProject_Add(whisper_cpp
    GIT_REPOSITORY https://github.com/ggml-org/whisper.cpp
    GIT_TAG        ${WHISPER_CPP_TAG}
    GIT_SHALLOW    TRUE
    CMAKE_ARGS     -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
                   -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON
    BUILD_COMMAND  ${CMAKE_COMMAND} --build <BINARY_DIR> --target whisper-cli --config Release
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy <BINARY_DIR>/bin/whisper-cli ${CMAKE_BINARY_DIR}/whisper-cli)
  add_dependencies(kibavoice whisper_cpp)
  install(PROGRAMS ${CMAKE_BINARY_DIR}/whisper-cli DESTINATION libexec/kibavoice)
endif()

# --- floating UI (Qt6 Widgets), a separate binary that drives the CLI -----
option(KIBAVOICE_BUILD_UI "Build kibavoice-ui (floating window)" ON)
if(KIBAVOICE_BUILD_UI)
  find_package(Qt6 COMPONENTS Widgets Network QUIET)
  if(Qt6_FOUND)
    add_subdirectory(ui)
    add_dependencies(kibavoice kibavoice-ui)
  else()
    message(WARNING "Qt6 (Widgets, Network) not found: skipping kibavoice-ui. "
                    "Install qt6-base-dev (Debian/Ubuntu) or qt6-base (Arch).")
  endif()
endif()

install(TARGETS kibavoice RUNTIME DESTINATION bin)
