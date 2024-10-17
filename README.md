# runtime-quad-test
Tests and examples of runtime quadrature use in FEniCSx (ffcx).
Compatible with ffcx version containing runtime quadrature integration kernel generation at ffcx fork https://github.com/sclaus2/ffcx 

Requires Basix 0.9.0 and later and ufl 2024.2.0. 

The runtime quadrature integration kernel can now be used via C++ and python and does not require a C-wrapper for Basix as tabulation is done outside of integral kernel. 

