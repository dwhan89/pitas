include ../compile_opts/$(CUSPS_COMP).mk

all: mcm_core.so

%.so: %.f90
	$(F2PY) --fcompiler=$(F2PYCOMP) --noopt -c ../wigner/wigner3j_sub.f -m $(basename $<) $< $(LAPACK_LINK) $(OMP_LINK) skip: map_border calc_weights

clean:
	rm -rf *.pyc *.so *.mod *.pyf
