# Find where we're running from, so we can store generated files here.
ifeq ($(origin MAKEFILE_DIR), undefined)
    MAKEFILE_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
endif
 
# Try to figure out the host system
HOST_OS :=
ifeq ($(OS),Windows_NT)
    HOST_OS = WINDOWS
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
            HOST_OS := LINUX
    endif
    ifeq ($(UNAME_S),Darwin)
        HOST_OS := OSX
    endif
endif
 
ARCH := $(shell if [[ $(shell uname -m) =~ i[345678]86 ]]; then echo x86_32; else echo $(shell uname -m); fi)
 
# Where compiled objects are stored.
OBJDIR := $(MAKEFILE_DIR)/gen/obj/
BINDIR := $(MAKEFILE_DIR)/gen/bin/
LIBDIR := $(MAKEFILE_DIR)/gen/lib/
GENDIR := $(MAKEFILE_DIR)/gen/obj/
 
# Settings for the host compiler.
CXX := $(CC_PREFIX) gcc
CXXFLAGS := --std=c++11 -O3 -DNDEBUG
CC := $(CC_PREFIX) gcc
CFLAGS :=
LDOPTS :=
LDOPTS += -L/usr/local/lib
ARFLAGS := -r
 
INCLUDES := \
-I. \
-I$(MAKEFILE_DIR)/../../../../../ \
-I$(MAKEFILE_DIR)/../../downloads/flatbuffers/include \
-I$(GENDIR)
# This is at the end so any globally-installed frameworks like protobuf don't
# override local versions in the source tree.
INCLUDES += -I/usr/local/include
 
LIBS := \
-lstdc++ \
-lpthread \
-lm \
-lz
 
# If we're on Linux, also link in the dl library.
ifeq ($(HOST_OS),LINUX)
    LIBS += -ldl -lpthread
endif
 
# This library is the main target for this makefile. It will contain a minimal
# runtime that can be linked in to other programs.
BIN_PATH := $(BINDIR)label_image
 
TF_LIB_PATH := $(MAKEFILE_DIR)/../../gen/lib/libtensorflow-lite.a
 
# What sources we want to compile, must be kept in sync with the main Bazel
# build files.
 
SRCS := \
tensorflow/contrib/lite/examples/label_image/label_image.cc \
tensorflow/contrib/lite/examples/label_image/bitmap_helpers.cc
 
OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(SRCS))))
 
# For normal manually-created TensorFlow C++ source files.
$(OBJDIR)%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
 
# For normal manually-created TensorFlow C++ source files.
$(OBJDIR)%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@
 
# The target that's compiled if there's no command-line arguments.
all: $(BIN_PATH)
 
$(BIN_PATH): $(OBJS) $(TF_LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(BIN_PATH) $(OBJS) \
	$(LIBFLAGS) $(TF_LIB_PATH) $(LDFLAGS) $(LIBS)
 
# Gets rid of all generated files.
clean:
	rm -rf $(MAKEFILE_DIR)/gen
 
# Gets rid of target files only, leaving the host alone. Also leaves the lib
# directory untouched deliberately, so we can persist multiple architectures
# across builds for iOS and Android.
cleantarget:
	rm -rf $(OBJDIR)
	rm -rf $(BINDIR)
 
$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d
 
-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(TF_CC_SRCS)))
 
 
 
 
