GCC = g++
GCC_FLAGS = -O2 -Wno-uninitialized -Wno-sign-compare -std=c++17 -I Header/
#-Wall -Wextra -Werror

OBJDIR = Object
MAINDIR = Main
MAINDIRTR = MainTR

OBJ = $(addprefix $(OBJDIR)/, $(patsubst Source/%.cc, %.o, $(wildcard Source/*.cc))) $(OBJDIR)/$(MAINDIR)/main.o
#OBJ_tr = $(addprefix $(OBJDIR)/, $(patsubst Source/%.cc, %.o, $(wildcard Source/*.cc))) $(OBJDIR)/$(MAINDIRTR)/main_tr.o

all: $(OBJDIR) $(OBJDIR)/$(MAINDIR) $(OBJDIR)/$(MAINDIRTR)  path_tracing_sec

$(OBJDIR):
	mkdir $(OBJDIR)

$(OBJDIR)/$(MAINDIR):
	mkdir $(OBJDIR)/$(MAINDIR)
	
#$(OBJDIR)/$(MAINDIRTR):
#	mkdir $(OBJDIR)/$(MAINDIRTR)

path_tracing_sec: $(OBJ)
	$(GCC) $(GCC_FLAGS) -o $@ $(OBJ)

#path_tracing_tr_sec: $(OBJ_tr)
#	$(GCC) $(GCC_FLAGS) -o $@ $(OBJ_tr)

$(OBJDIR)/$(MAINDIR)/%.o: %.cc
	$(GCC) $(GCC_FLAGS) -o $@ -c $<
	
$(OBJDIR)/$(MAINDIRTR)/%.o: %.cc
	$(GCC) $(GCC_FLAGS) -o $@ -c $<

$(OBJDIR)/%.o: Source/%.cc Header/%.hh
	$(GCC) $(GCC_FLAGS) -o $@ -c $<

projecte.tar:
	tar -cvf $@  Source/*.cc Header/*.hh main.cc Makefile
	
clean:
	rm -rf $(OBJDIR)/*.o $(OBJDIR)/$(MAINDIR)/*.o $(OBJDIR)/$(MAINDIRTR)/*.o  path_tracing_sec *.ppm
	



