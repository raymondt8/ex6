# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/raymond/Documents/TMA4280/ex6_2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/raymond/Documents/TMA4280/ex6_2/build

# Include any dependencies generated for this target.
include CMakeFiles/poisson.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/poisson.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/poisson.dir/flags.make

CMakeFiles/poisson.dir/poissonMain.c.o: CMakeFiles/poisson.dir/flags.make
CMakeFiles/poisson.dir/poissonMain.c.o: ../poissonMain.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/raymond/Documents/TMA4280/ex6_2/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/poisson.dir/poissonMain.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/poisson.dir/poissonMain.c.o   -c /home/raymond/Documents/TMA4280/ex6_2/poissonMain.c

CMakeFiles/poisson.dir/poissonMain.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/poisson.dir/poissonMain.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/raymond/Documents/TMA4280/ex6_2/poissonMain.c > CMakeFiles/poisson.dir/poissonMain.c.i

CMakeFiles/poisson.dir/poissonMain.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/poisson.dir/poissonMain.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/raymond/Documents/TMA4280/ex6_2/poissonMain.c -o CMakeFiles/poisson.dir/poissonMain.c.s

CMakeFiles/poisson.dir/poissonMain.c.o.requires:
.PHONY : CMakeFiles/poisson.dir/poissonMain.c.o.requires

CMakeFiles/poisson.dir/poissonMain.c.o.provides: CMakeFiles/poisson.dir/poissonMain.c.o.requires
	$(MAKE) -f CMakeFiles/poisson.dir/build.make CMakeFiles/poisson.dir/poissonMain.c.o.provides.build
.PHONY : CMakeFiles/poisson.dir/poissonMain.c.o.provides

CMakeFiles/poisson.dir/poissonMain.c.o.provides.build: CMakeFiles/poisson.dir/poissonMain.c.o

CMakeFiles/poisson.dir/poissonFunctions.c.o: CMakeFiles/poisson.dir/flags.make
CMakeFiles/poisson.dir/poissonFunctions.c.o: ../poissonFunctions.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/raymond/Documents/TMA4280/ex6_2/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/poisson.dir/poissonFunctions.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/poisson.dir/poissonFunctions.c.o   -c /home/raymond/Documents/TMA4280/ex6_2/poissonFunctions.c

CMakeFiles/poisson.dir/poissonFunctions.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/poisson.dir/poissonFunctions.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/raymond/Documents/TMA4280/ex6_2/poissonFunctions.c > CMakeFiles/poisson.dir/poissonFunctions.c.i

CMakeFiles/poisson.dir/poissonFunctions.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/poisson.dir/poissonFunctions.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/raymond/Documents/TMA4280/ex6_2/poissonFunctions.c -o CMakeFiles/poisson.dir/poissonFunctions.c.s

CMakeFiles/poisson.dir/poissonFunctions.c.o.requires:
.PHONY : CMakeFiles/poisson.dir/poissonFunctions.c.o.requires

CMakeFiles/poisson.dir/poissonFunctions.c.o.provides: CMakeFiles/poisson.dir/poissonFunctions.c.o.requires
	$(MAKE) -f CMakeFiles/poisson.dir/build.make CMakeFiles/poisson.dir/poissonFunctions.c.o.provides.build
.PHONY : CMakeFiles/poisson.dir/poissonFunctions.c.o.provides

CMakeFiles/poisson.dir/poissonFunctions.c.o.provides.build: CMakeFiles/poisson.dir/poissonFunctions.c.o

# Object files for target poisson
poisson_OBJECTS = \
"CMakeFiles/poisson.dir/poissonMain.c.o" \
"CMakeFiles/poisson.dir/poissonFunctions.c.o"

# External object files for target poisson
poisson_EXTERNAL_OBJECTS =

poisson: CMakeFiles/poisson.dir/poissonMain.c.o
poisson: CMakeFiles/poisson.dir/poissonFunctions.c.o
poisson: CMakeFiles/poisson.dir/build.make
poisson: libcommon.a
poisson: /usr/lib/x86_64-linux-gnu/libmpich.so
poisson: /usr/lib/x86_64-linux-gnu/libopa.so
poisson: /usr/lib/x86_64-linux-gnu/libmpl.so
poisson: /usr/lib/x86_64-linux-gnu/librt.so
poisson: /usr/lib/libcr.so
poisson: /usr/lib/x86_64-linux-gnu/libpthread.so
poisson: CMakeFiles/poisson.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking C executable poisson"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/poisson.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/poisson.dir/build: poisson
.PHONY : CMakeFiles/poisson.dir/build

CMakeFiles/poisson.dir/requires: CMakeFiles/poisson.dir/poissonMain.c.o.requires
CMakeFiles/poisson.dir/requires: CMakeFiles/poisson.dir/poissonFunctions.c.o.requires
.PHONY : CMakeFiles/poisson.dir/requires

CMakeFiles/poisson.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/poisson.dir/cmake_clean.cmake
.PHONY : CMakeFiles/poisson.dir/clean

CMakeFiles/poisson.dir/depend:
	cd /home/raymond/Documents/TMA4280/ex6_2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/raymond/Documents/TMA4280/ex6_2 /home/raymond/Documents/TMA4280/ex6_2 /home/raymond/Documents/TMA4280/ex6_2/build /home/raymond/Documents/TMA4280/ex6_2/build /home/raymond/Documents/TMA4280/ex6_2/build/CMakeFiles/poisson.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/poisson.dir/depend

