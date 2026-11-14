cmake_minimum_required(VERSION 3.18)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(kibavoice-ui LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 COMPONENTS Widgets Network REQUIRED)

add_executable(kibavoice-ui
  main.cpp window.cpp widgets.cpp audio.cpp backend.cpp resources.qrc)
target_link_libraries(kibavoice-ui PRIVATE Qt6::Widgets Qt6::Network)

install(TARGETS kibavoice-ui RUNTIME DESTINATION bin)
install(FILES assets/kibavoice.png DESTINATION share/kibavoice)
