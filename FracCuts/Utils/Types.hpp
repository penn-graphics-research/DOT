//
//  Type.hpp
//  FracCuts
//
//  Created by Minchen Li on 2/5/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef Types_hpp
#define Types_hpp

#include <cstdio>

//#define LINSYSSOLVER_USE_CHOLMOD
//#define LINSYSSOLVER_USE_PARDISO

//#define USE_TBB

//#define USE_METIS

#define USE_CLOSEDFORMSVD2D

namespace FracCuts {
    enum MethodType {
        MT_OURS,
        MT_AUTOCUTS,
        MT_GEOMIMG,
        MT_OURS_FIXED,
        MT_NOCUT
    };
}

#endif /* Types_hpp */
