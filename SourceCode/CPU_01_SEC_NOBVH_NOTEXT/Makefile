GCC = g++
GCC_FLAGS = -O2 -Wno-uninitialized -Wno-sign-compare -std=c++17 -I Header/

OBJDIR = Object
MAINDIR = Main

OBJ = $(addprefix $(OBJDIR)/, $(patsubst Source/%.cc, %.o, $(wildcard Source/*.cc))) $(OBJDIR)/$(MAINDIR)/main.o

all: $(OBJDIR) $(OBJDIR)/$(MAINDIR) path_tracing_sec

$(OBJDIR):
	mkdir $(OBJDIR)

$(MAINDIR):
	mkdir $(OBJDIR)/$(MAINDIR)

path_tracing_sec: $(OBJ)
	$(GCC) $(GCC_FLAGS) -o $@ $(OBJ)

$(OBJDIR)/$(MAINDIR)/%.o: %.cc
	$(GCC) $(GCC_FLAGS) -o $@ -c $<

$(OBJDIR)/%.o: Source/%.cc Header/%.hh
	$(GCC) $(GCC_FLAGS) -o $@ -c $<

projecte.tar:
	tar -cvf $@  Source/*.cc Header/*.hh Makefile
	
clean:
	rm -rf $(OBJDIR)/*.o $(OBJDIR)/$(MAINDIR)/*.o path_tracing_sec *.ppm
	



