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

TESTSRCS = $(TESTDIR)/test_micro_nn.c $(TESTDIR)/test_logistic.c \
           $(TESTDIR)/test_kmeans.c $(TESTDIR)/test_isolation.c \
           $(TESTDIR)/test_main.c

LIB     = libexocortex.a
TESTBIN = test_runner
ASANBIN = $(TESTBIN)_asan

.PHONY: all clean test valgrind lint asan

all: $(LIB)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): $(OBJS)
	ar rcs $@ $^

test: $(TESTBIN)
	./$(TESTBIN)

$(TESTBIN): $(OBJS) $(TESTSRCS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(TESTSRCS) $(LDFLAGS)

valgrind: $(TESTBIN)
	valgrind --leak-check=full --error-exitcode=1 ./$(TESTBIN)

lint:
	$(CC) $(CFLAGS) -Werror -fsyntax-only $(SRCS)

asan: $(ASANBIN)
	./$(ASANBIN)

$(ASANBIN): $(SRCS) $(TESTSRCS)
	$(CC) $(CFLAGS) -O1 -g -fsanitize=address,undefined -o $@ $(SRCS) $(TESTSRCS) $(LDFLAGS)

clean:
	rm -rf $(OBJDIR) $(LIB) $(TESTBIN) $(ASANBIN)
