# PITAS
**P**ower-spectrum **I**n **T**racts **A**lgorithm on the **S**phere

------
PITAS is an agile routine to compute power spectrum of curved-sky in cylidrical projection (CAR, CEA). The core of the routine is the *MASTER* algorithm implemented by Thibaut Louis. And it contains an implementation of *PURE-EB* to compute unbiased CMB polarization spectra. As a part of package, it contains DR3JJ routine to compute Wigner 3j symbols.  

------

![alt text](https://github.com/dwhan89/pitas/blob/master/resource/pita2.jpg "PITAS")
 

------

## Installation

### Core dependencies
1. **pixell** (https://github.com/simonsobs/pixell)
* In particular, PITAS requires curvedsky module.
2. your favorite FORTRAN compiler to compile PITAS!

### Installation steps
```
  cd <path to pitas>/pitas
  export PITAS_COMP=<Compiler File> ! ex) export PITAS_COMP=nersc_cori
                                    ! Check pitas/compile_opts for more options
  make
  cd ..
  pip install -e . --user           ! install python module 
```

------
## Misc/Note
For a trial run, please take a look at ```test/simple_test.py```.


