CC = gcc
CFLAGS = -Wall -ggdb3 -O5
LDFLAGS = -L. -lm

all: libenergy.so

libenergy.so: energy.c
	$(CC) $(CFLAGS) -shared -o libenergy.so -fPIC energy.c


clean: FORCE
	@-rm libenergy.so

FORCE:
