# PITAS
**P**ower-spectrum **I**n **T**racts **A**lgorithm on the **S**phere

**"Best Unbiased Pita on the Sphere™"** 

------
PITAS is an agile routine to compute power spectrum of curved-sky in cylidrical projection (CAR, CEA). The core of the routine is the *MASTER* algorithm implemented by Thibaut Louis. And it contains an implementation of *PURE-EB* to compute unbiased CMB polarization spectra. As a part of package, it contains DR3JJ routine to compute Wigner 3j symbols.   

------

## Installation

### Core dependencies
1. **libsharp** (https://github.com/Libsharp/libsharp)
* used for Spherical Harmronic Transform (SHT)
2. **enlib** (https://github.com/amaurea/enlib)
* In particular, PITAS requires curvedsky module.
3. your favorite FORTRAN compiler to compile PITAS!

### Installation steps
```
  cd <path to pitas>/pitas
  export PITAS_COMP=<Compiler File> ! ex) export PITAS_COMP=nersc_cori
                                    ! Check pitas/compile_opts for all options
  make
  cd ..
  pip install -e . --user           ! install python module 
```

------
## Misc/Note
For a trial run, please take a look at ```test/script/simple_test.py```.


