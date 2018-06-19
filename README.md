# CUSPS
**CU**rved **S**ky **P**ower **S**pectra

------
This code is a wrapper around Thibaut's curved sky mode coupling routine, so that it can seamlessly integrate with existing ACT analysis pipelines. 

------

## Installation

### Core dependencies
1. **libsharp** (https://github.com/Libsharp/libsharp)
* used for Spherical Harmronic Transform (SHT)
2. **enlib** (https://github.com/amaurea/enlib)
* In particular, CUSPS requires curvedsky module.
3. your favorite FORTRAN compiler to compile CUSPS!

### Installation steps
```
  cd <path to cusps>/cusps
  export CUSPS_COMP=<Compiler File> ! ex) export CUSPS_COMP=nersc_cori
                                    ! Check cusps/compile_opts for all options
  make
  cd ..
  pip install -e . --user 
```

------
## Misc/Note
For a trial run, please take a look at ```test/script/simple_test.py```.


