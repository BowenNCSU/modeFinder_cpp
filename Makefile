# Build the modeFinder C library and the Python extension.
#
#   make            shared library libmodefinder.{so,dylib} and static libmodefinder.a
#   make example    build and run examples/example.c against the static library
#   make python     build the Python extension in place (python -m pip install . also works)
#   make test       run the Python tests
#   make clean

CC      ?= cc
CFLAGS  ?= -O2
CFLAGS  += -std=c11 -Wall -Wextra -Wno-unused-parameter -fPIC -Icsrc
LDLIBS  += -lm
PYTHON  ?= python3

SRCS = csrc/modefinder.c csrc/kde.c csrc/lbfgsb.c csrc/linpack.c
OBJS = $(SRCS:.c=.o)

ifeq ($(shell uname -s),Darwin)
  SHLIB = libmodefinder.dylib
  SHFLAGS = -dynamiclib -install_name @rpath/$(SHLIB)
else
  SHLIB = libmodefinder.so
  SHFLAGS = -shared
endif

all: $(SHLIB) libmodefinder.a

$(OBJS): csrc/modefinder.h csrc/mf_internal.h

$(SHLIB): $(OBJS)
	$(CC) $(SHFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS)

libmodefinder.a: $(OBJS)
	$(AR) rcs $@ $(OBJS)

examples/example: examples/example.c libmodefinder.a
	$(CC) $(CFLAGS) -o $@ $< libmodefinder.a $(LDFLAGS) $(LDLIBS)

example: examples/example
	./examples/example

python:
	$(PYTHON) setup.py build_ext --inplace

test: python
	$(PYTHON) -m unittest discover -s tests -v

clean:
	rm -rf $(OBJS) $(SHLIB) libmodefinder.a examples/example build *.egg-info \
	       modefinder/*.so modefinder/*.pyd modefinder/__pycache__ tests/__pycache__

.PHONY: all example python test clean
