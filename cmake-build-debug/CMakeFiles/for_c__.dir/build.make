# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.23

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\JetBrains\CLion 2022.2.3\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\JetBrains\CLion 2022.2.3\bin\cmake\win\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\ximen\Desktop\Use-C--Play-Neural-Network

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\ximen\Desktop\Use-C--Play-Neural-Network\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/for_c__.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/for_c__.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/for_c__.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/for_c__.dir/flags.make

CMakeFiles/for_c__.dir/new_version_nn.cpp.obj: CMakeFiles/for_c__.dir/flags.make
CMakeFiles/for_c__.dir/new_version_nn.cpp.obj: ../new_version_nn.cpp
CMakeFiles/for_c__.dir/new_version_nn.cpp.obj: CMakeFiles/for_c__.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\ximen\Desktop\Use-C--Play-Neural-Network\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/for_c__.dir/new_version_nn.cpp.obj"
	C:\PROGRA~1\JETBRA~1\CLION2~1.3\bin\mingw\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/for_c__.dir/new_version_nn.cpp.obj -MF CMakeFiles\for_c__.dir\new_version_nn.cpp.obj.d -o CMakeFiles\for_c__.dir\new_version_nn.cpp.obj -c C:\Users\ximen\Desktop\Use-C--Play-Neural-Network\new_version_nn.cpp

CMakeFiles/for_c__.dir/new_version_nn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/for_c__.dir/new_version_nn.cpp.i"
	C:\PROGRA~1\JETBRA~1\CLION2~1.3\bin\mingw\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\ximen\Desktop\Use-C--Play-Neural-Network\new_version_nn.cpp > CMakeFiles\for_c__.dir\new_version_nn.cpp.i

CMakeFiles/for_c__.dir/new_version_nn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/for_c__.dir/new_version_nn.cpp.s"
	C:\PROGRA~1\JETBRA~1\CLION2~1.3\bin\mingw\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\ximen\Desktop\Use-C--Play-Neural-Network\new_version_nn.cpp -o CMakeFiles\for_c__.dir\new_version_nn.cpp.s

# Object files for target for_c__
for_c___OBJECTS = \
"CMakeFiles/for_c__.dir/new_version_nn.cpp.obj"

# External object files for target for_c__
for_c___EXTERNAL_OBJECTS =

for_c__.exe: CMakeFiles/for_c__.dir/new_version_nn.cpp.obj
for_c__.exe: CMakeFiles/for_c__.dir/build.make
for_c__.exe: CMakeFiles/for_c__.dir/linklibs.rsp
for_c__.exe: CMakeFiles/for_c__.dir/objects1.rsp
for_c__.exe: CMakeFiles/for_c__.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\ximen\Desktop\Use-C--Play-Neural-Network\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable for_c__.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\for_c__.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/for_c__.dir/build: for_c__.exe
.PHONY : CMakeFiles/for_c__.dir/build

CMakeFiles/for_c__.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\for_c__.dir\cmake_clean.cmake
.PHONY : CMakeFiles/for_c__.dir/clean

CMakeFiles/for_c__.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\ximen\Desktop\Use-C--Play-Neural-Network C:\Users\ximen\Desktop\Use-C--Play-Neural-Network C:\Users\ximen\Desktop\Use-C--Play-Neural-Network\cmake-build-debug C:\Users\ximen\Desktop\Use-C--Play-Neural-Network\cmake-build-debug C:\Users\ximen\Desktop\Use-C--Play-Neural-Network\cmake-build-debug\CMakeFiles\for_c__.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/for_c__.dir/depend

