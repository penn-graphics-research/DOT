//
//  METIS.hpp
//  FracCuts
//
//  Created by Minchen Li on 8/3/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef METIS_hpp
#define METIS_hpp

#include "TriangleSoup.hpp"

#include <metis.h>

namespace FracCuts {
    
    // data structure from METIS mesh partitioning example
    typedef struct {
        idx_t ptype;
        idx_t objtype;
        idx_t ctype;
        idx_t iptype;
        idx_t rtype;
        
        idx_t no2hop;
        idx_t minconn;
        idx_t contig;
        
        idx_t nooutput;
        
        idx_t balance;
        idx_t ncuts;
        idx_t niter;
        
        idx_t gtype;
        idx_t ncommon;
        
        idx_t seed;
        idx_t dbglvl;
        
        idx_t nparts;
        
        idx_t nseps;
        idx_t ufactor;
        idx_t pfactor;
        idx_t compress;
        idx_t ccorder;
        
        char *filename;
        char *outfile;
        char *xyzfile;
        char *tpwgtsfile;
        char *ubvecstr;
        
        idx_t wgtflag;
        idx_t numflag;
        real_t *tpwgts;
        real_t *ubvec;
        
        real_t iotimer;
        real_t parttimer;
        real_t reporttimer;
        
        size_t maxmemory;
    } params_t;
    
    class METIS
    {
    protected: // data
        idx_t nn, ne; // METIS mesh info
        std::vector<idx_t> eptr, eind; // METIS mesh data structure
        std::vector<idx_t> epart, npart; // output partition info
        
    public: // constructor
        METIS(const TriangleSoup& mesh);
        
    public: // API
        void partMesh(idx_t nparts);
        void getElementList(int partI, Eigen::VectorXi& elementListI) const;
        
    protected: // helper
        void initParam(params_t& params, idx_t nparts);
        
        void printInput(const idx_t *ewgt,
                        const params_t& params) const;
        void printOutput(void) const;
    };
    
}

#endif /* METIS_hpp */
