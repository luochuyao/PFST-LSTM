# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ices/PycharmProject/HKO/VarFlow

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ices/PycharmProject/HKO/VarFlow/build

# Include any dependencies generated for this target.
include CMakeFiles/FlowExample.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/FlowExample.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/FlowExample.dir/flags.make

CMakeFiles/FlowExample.dir/example.cpp.o: CMakeFiles/FlowExample.dir/flags.make
CMakeFiles/FlowExample.dir/example.cpp.o: ../example.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ices/PycharmProject/HKO/VarFlow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/FlowExample.dir/example.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FlowExample.dir/example.cpp.o -c /home/ices/PycharmProject/HKO/VarFlow/example.cpp

CMakeFiles/FlowExample.dir/example.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FlowExample.dir/example.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ices/PycharmProject/HKO/VarFlow/example.cpp > CMakeFiles/FlowExample.dir/example.cpp.i

CMakeFiles/FlowExample.dir/example.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FlowExample.dir/example.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ices/PycharmProject/HKO/VarFlow/example.cpp -o CMakeFiles/FlowExample.dir/example.cpp.s

CMakeFiles/FlowExample.dir/VarFlow.cpp.o: CMakeFiles/FlowExample.dir/flags.make
CMakeFiles/FlowExample.dir/VarFlow.cpp.o: ../VarFlow.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ices/PycharmProject/HKO/VarFlow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/FlowExample.dir/VarFlow.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FlowExample.dir/VarFlow.cpp.o -c /home/ices/PycharmProject/HKO/VarFlow/VarFlow.cpp

CMakeFiles/FlowExample.dir/VarFlow.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FlowExample.dir/VarFlow.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ices/PycharmProject/HKO/VarFlow/VarFlow.cpp > CMakeFiles/FlowExample.dir/VarFlow.cpp.i

CMakeFiles/FlowExample.dir/VarFlow.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FlowExample.dir/VarFlow.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ices/PycharmProject/HKO/VarFlow/VarFlow.cpp -o CMakeFiles/FlowExample.dir/VarFlow.cpp.s

# Object files for target FlowExample
FlowExample_OBJECTS = \
"CMakeFiles/FlowExample.dir/example.cpp.o" \
"CMakeFiles/FlowExample.dir/VarFlow.cpp.o"

# External object files for target FlowExample
FlowExample_EXTERNAL_OBJECTS =

FlowExample: CMakeFiles/FlowExample.dir/example.cpp.o
FlowExample: CMakeFiles/FlowExample.dir/VarFlow.cpp.o
FlowExample: CMakeFiles/FlowExample.dir/build.make
FlowExample: /home/ices/anaconda3/lib/libopencv_xphoto.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_xobjdetect.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_tracking.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_surface_matching.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_structured_light.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_stereo.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_saliency.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_rgbd.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_reg.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_plot.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_optflow.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_line_descriptor.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_fuzzy.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_dpm.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_dnn.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_datasets.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_ccalib.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_bioinspired.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_bgsegm.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_aruco.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_videostab.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_superres.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_stitching.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_photo.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_text.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_face.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_ximgproc.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_xfeatures2d.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_shape.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_video.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_objdetect.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_calib3d.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_features2d.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_ml.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_highgui.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_videoio.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_imgcodecs.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_imgproc.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_flann.so.3.1.0
FlowExample: /home/ices/anaconda3/lib/libopencv_core.so.3.1.0
FlowExample: CMakeFiles/FlowExample.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ices/PycharmProject/HKO/VarFlow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable FlowExample"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FlowExample.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/FlowExample.dir/build: FlowExample

.PHONY : CMakeFiles/FlowExample.dir/build

CMakeFiles/FlowExample.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/FlowExample.dir/cmake_clean.cmake
.PHONY : CMakeFiles/FlowExample.dir/clean

CMakeFiles/FlowExample.dir/depend:
	cd /home/ices/PycharmProject/HKO/VarFlow/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ices/PycharmProject/HKO/VarFlow /home/ices/PycharmProject/HKO/VarFlow /home/ices/PycharmProject/HKO/VarFlow/build /home/ices/PycharmProject/HKO/VarFlow/build /home/ices/PycharmProject/HKO/VarFlow/build/CMakeFiles/FlowExample.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/FlowExample.dir/depend
