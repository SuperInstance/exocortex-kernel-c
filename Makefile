CC      = gcc
CFLAGS  = -std=c11 -Wall -Wextra -Wpedantic -O2 -I. -Iinclude
LDFLAGS = -lm
SRCDIR  = src
OBJDIR  = obj
TESTDIR = tests

SRCS    = $(SRCDIR)/matrix.c $(SRCDIR)/stats.c $(SRCDIR)/micro_nn.c \
          $(SRCDIR)/logistic.c $(SRCDIR)/kmeans.c $(SRCDIR)/isolation.c \
          $(SRCDIR)/compute.c
OBJS    = $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(SRCS))

LIB     = libexocortex.a
TESTBIN = test_runner

.PHONY: all clean test valgrind

all: $(LIB)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): $(OBJS)
	ar rcs $@ $^

test: $(TESTBIN)
	./$(TESTBIN)

$(TESTBIN): $(OBJS) $(TESTDIR)/test_micro_nn.c $(TESTDIR)/test_logistic.c \
            $(TESTDIR)/test_kmeans.c $(TESTDIR)/test_isolation.c \
            $(TESTDIR)/test_main.c
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(TESTDIR)/test_micro_nn.c \
	    $(TESTDIR)/test_logistic.c $(TESTDIR)/test_kmeans.c \
	    $(TESTDIR)/test_isolation.c $(TESTDIR)/test_main.c $(LDFLAGS)

valgrind: $(TESTBIN)
	valgrind --leak-check=full --error-exitcode=1 ./$(TESTBIN)

clean:
	rm -rf $(OBJDIR) $(LIB) $(TESTBIN)
