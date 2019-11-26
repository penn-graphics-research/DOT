//
//  Type.hpp
//  DOT
//
//  Created by Minchen Li on 2/5/18.
//

#ifndef Types_hpp
#define Types_hpp

#include <cstdio>

#define LINSYSSOLVER_USE_CHOLMOD
//#define LINSYSSOLVER_USE_PARDISO

#define USE_TBB

#define USE_METIS 1

#define USE_GW

#define OVERBYAPD
#define SVSPACE_FSTEP
//#define STRICTPPD

#define DIM 3

#define USE_IQRSVD

#if(DIM == 3)
#define USE_SIMD
#endif

#define ALPHAINIT

//#define SNH_WITHLOG
#if defined(SNH_WITHLOG) && defined(USE_SIMD)
#undef USE_SIMD
#endif

#include "Triplet.h"

#endif /* Types_hpp */
