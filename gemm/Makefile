
BLIS_PREFIX := $(HOME)/local

output  := a.out
objects := main.o

LD       := $(CXX)
CPPFLAGS := 
CXXFLAGS := -g3 -O3 -std=c++11 -march=native -fopenmp
LDFLAGS  := -fopenmp

# for Intel compiler
ifeq ($(AVX512),1)
  CXXFLAGS += -DUSE_AVX512
endif

ifeq ($(BLIS),1)
  CPPFLAGS += -I$(BLIS_PREFIX)/include -DUSE_BLIS
  LDFLAGS  += -L$(BLIS_PREFIX)/lib
  LDLIBS   += -lblis -lmemkind
endif

ifeq ($(MKL),1)
  CPPFLAGS += -DUSE_MKL
  LDLIBS   += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
endif

$(output) : $(objects)
	$(LD) $(LDFLAGS) $^ -o $@ $(LDLIBS)

.PHONY: clean

clean:
	$(RM) $(objects) $(output)

