# Uniformly-Partitioned-Convolution
Time-domain partitioned FIR convolution with overlap-add (educational implementation).
This repository contains a MATLAB implementation of partitioned convolution in the time domain, following the derivation


The code explicitly demonstrates:

uniform partitioning of the impulse response

ring buffer management for input blocks

overlap-add to ensure equivalence with linear convolution

Verified against MATLAB filter with negligible numerical error.
