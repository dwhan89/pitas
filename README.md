# PITAS
**P**ower-spectrum **I**n **T**racts **A**lgorithm on the **S**phere

------
PITAS is an agile routine to compute power spectra of curved-sky in cylidrical projection (CAR, CEA). The core of the routine is the *MASTER* algorithm implemented by Thibaut Louis. It implements *PURE-EB* to compute unbiased CMB polarization spectra. DRC3JJ routine is packaged together to compute Wigner 3j symbol.  

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


