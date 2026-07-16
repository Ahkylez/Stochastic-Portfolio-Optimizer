cmake_minimum_required(VERSION 3.18)
project(MyOrToolsProject LANGUAGES CXX)

# OR-Tools requires C++17 or newer
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 1. OR-Tools location -- configurable so the build is portable.
#    Priority:  -DOR_TOOLS_DIR=...   >   $ENV{OR_TOOLS_DIR}   >   per-OS default.
if(NOT OR_TOOLS_DIR)
    if(DEFINED ENV{OR_TOOLS_DIR})
        set(OR_TOOLS_DIR "$ENV{OR_TOOLS_DIR}")
    elseif(WIN32)
        set(OR_TOOLS_DIR "C:/Users/willi/Libraries/or-tools")
    else()
        set(OR_TOOLS_DIR "$ENV{HOME}/libs/or-tools")
    endif()
endif()
set(OR_TOOLS_DIR "${OR_TOOLS_DIR}" CACHE PATH "Path to the OR-Tools installation root")
list(APPEND CMAKE_PREFIX_PATH ${OR_TOOLS_DIR})

# 2. Find OR-Tools
find_package(ortools CONFIG REQUIRED PATHS ${OR_TOOLS_DIR})

# 3. Find OpenMP for parallel evaluation
find_package(OpenMP REQUIRED)

# 4. Define the executable
add_executable(optimizer_app
    src/pbil.cpp
    src/hybrid_pbil.cpp
)

# 5. Link libraries
target_link_libraries(optimizer_app PRIVATE
    ortools::ortools
    OpenMP::OpenMP_CXX
)

# 6. Compiler-specific options
if(MSVC)
    # Increase stack size to 64 MB (default 1 MB is too small for SCIP's recursive B&B)
    target_link_options(optimizer_app PRIVATE /STACK:67108864)

    # Enable proper __cplusplus reporting and faster builds
    target_compile_options(optimizer_app PRIVATE
        /Zc:__cplusplus
        /MP                # parallel compilation
        /EHsc              # standard exception handling
    )

    # Release-mode optimizations
    target_compile_options(optimizer_app PRIVATE
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:RelWithDebInfo>:/O2>
    )
else()
    # GCC / Clang (Linux, WSL, macOS)
    target_compile_options(optimizer_app PRIVATE
        $<$<CONFIG:Release>:-O2>
        $<$<CONFIG:RelWithDebInfo>:-O2>
    )
    # SCIP's recursive branch-and-bound needs a large stack. On Linux there is no
    # link-time stack flag; raise it at runtime instead:
    #   ulimit -s unlimited            # main thread
    #   export OMP_STACKSIZE=64M       # OpenMP worker threads
endif()

# 7. Runtime lookup of the OR-Tools shared libraries.
if(WIN32)
    # Windows: copy the OR-Tools DLLs next to the .exe after building.
    add_custom_command(TARGET optimizer_app POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${OR_TOOLS_DIR}/bin"
            "$<TARGET_FILE_DIR:optimizer_app>"
    )
else()
    # Linux / macOS: bake the OR-Tools lib dir into the binary's rpath so it runs
    # without needing LD_LIBRARY_PATH to be set by hand.
    set_target_properties(optimizer_app PROPERTIES
        BUILD_RPATH   "${OR_TOOLS_DIR}/lib"
        INSTALL_RPATH "${OR_TOOLS_DIR}/lib"
    )
endif()

# 8. Sanity output
message(STATUS "OR-Tools found at: ${OR_TOOLS_DIR}")
message(STATUS "OpenMP CXX flags: ${OpenMP_CXX_FLAGS}")
