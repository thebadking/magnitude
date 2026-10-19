// One channel per ordered device queue. A failed channel is never reused:
// later launches must not overwrite input a late host consumer still reads.
#include "control.cuh"
