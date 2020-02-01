GCC = g++
GCC_FLAGS = -O2 -Wno-uninitialized -Wno-sign-compare -std=c++17 -I Header/

OMP = -fopenmp

OBJDIR = Object
MAINDIR = Main

OBJ = $(addprefix $(OBJDIR)/, $(patsubst Source/%.cc, %.o, $(wildcard Source/*.cc))) $(OBJDIR)/$(MAINDIR)/main.o

all: $(OBJDIR) $(OBJDIR)/$(MAINDIR) $(OBJDIR)/$(MAINDIRTR)  path_tracing_sec

$(OBJDIR):
	mkdir $(OBJDIR)

$(OBJDIR)/$(MAINDIR):
	mkdir $(OBJDIR)/$(MAINDIR)

path_tracing_sec: $(OBJ)
	$(GCC) $(GCC_FLAGS) -o $@ $(OBJ) $(OMP)

$(OBJDIR)/$(MAINDIR)/%.o: %.cc
	$(GCC) $(GCC_FLAGS) -o $@ -c $< $(OMP)

$(OBJDIR)/$(MAINDIRTR)/%.o: %.cc
	$(GCC) $(GCC_FLAGS) -o $@ -c $<

$(OBJDIR)/%.o: Source/%.cc Header/%.hh
	$(GCC) $(GCC_FLAGS) -o $@ -c $<

projecte.tar:
	tar -cvf $@  Source/*.cc Header/*.hh main.cc Makefile
	
clean:
	rm -rf $(OBJDIR)/*.o $(OBJDIR)/$(MAINDIR)/*.o $(OBJDIR)/$(MAINDIRTR)/*.o  path_tracing_sec *.ppm
	



