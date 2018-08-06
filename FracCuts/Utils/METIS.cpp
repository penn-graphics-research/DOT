//
//  METIS.cpp
//  FracCuts
//
//  Created by Minchen Li on 8/3/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#include "METIS.hpp"

#include <iostream>

namespace FracCuts {
    
    METIS::METIS(const TriangleSoup& mesh)
    {
        // construct METIS mesh representation
        eptr.resize(mesh.F.rows() + 1);
        eind.resize(mesh.F.rows() * 3);
        for(int triI = 0; triI < mesh.F.rows(); triI++) {
            eptr[triI] = triI * 3;
            const Eigen::RowVector3i& triVInd = mesh.F.row(triI);
            for(int i = 0; i < 3; i++) {
                eind[eptr[triI] + i] = triVInd[i];
            }
        }
        eptr[mesh.F.rows()] = mesh.F.rows() * 3;
        
        ne = mesh.F.rows();
        nn = mesh.V_rest.rows();
    }
    
    void METIS::partMesh(idx_t nparts)
    {
        // init params and options
        params_t params;
        initParam(params, nparts);
        
        idx_t options[METIS_NOPTIONS];
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_PTYPE]   = params.ptype;
        options[METIS_OPTION_OBJTYPE] = params.objtype;
        options[METIS_OPTION_CTYPE]   = params.ctype;
        options[METIS_OPTION_IPTYPE]  = params.iptype;
        options[METIS_OPTION_RTYPE]   = params.rtype;
        options[METIS_OPTION_DBGLVL]  = params.dbglvl;
        options[METIS_OPTION_UFACTOR] = params.ufactor;
        options[METIS_OPTION_MINCONN] = params.minconn;
        options[METIS_OPTION_CONTIG]  = params.contig;
        options[METIS_OPTION_SEED]    = params.seed;
        options[METIS_OPTION_NITER]   = params.niter;
        options[METIS_OPTION_NCUTS]   = params.ncuts;
        
        // init other parameters
        epart.resize(ne);
        npart.resize(nn);
        std::vector<idx_t> ewgt(ne, 1); // element weights, computational cost
        std::vector<real_t> tpwgts(nparts, 1.0 / nparts); // part weights
        idx_t objval;
        
        int status = METIS_PartMeshDual(&ne, &nn, eptr.data(), eind.data(),
                                        ewgt.data(), NULL, // communication cost
                                        &params.ncommon, &params.nparts,
                                        tpwgts.data(), options, &objval, epart.data(), npart.data());
        
        switch(status) {
            case METIS_OK:
                std::cout << "METIS OK" << std::endl;
                break;

            case METIS_ERROR_INPUT:
                std::cout << "METIS ERROR: erroneous inputs and/or options" << std::endl;
                break;

            case METIS_ERROR_MEMORY:
                std::cout << "METIS ERROR: insufficient memory" << std::endl;
                break;

            case METIS_ERROR:
                std::cout << "METIS ERROR: other errors" << std::endl;
                break;
        }
    }
    
    void METIS::getElementList(int partI, Eigen::VectorXi& elementListI) const
    {
        assert(!epart.empty());
        
        // load element list
        elementListI.resize(0);
        for(int triI = 0; triI < epart.size(); triI++) {
            if(epart[triI] == partI) {
                int oldSize = elementListI.size();
                elementListI.conservativeResize(oldSize + 1);
                elementListI[oldSize] = triI;
            }
        }
    }
    
    
    void METIS::initParam(params_t& params, idx_t nparts)
    {
        memset((void *)&params, 0, sizeof(params_t));
        
        /* initialize the params data structure */
        params.gtype         = METIS_GTYPE_DUAL;
        params.ptype         = METIS_PTYPE_KWAY;
        params.objtype       = METIS_OBJTYPE_CUT;
        params.ctype         = METIS_CTYPE_SHEM;
        params.iptype        = METIS_IPTYPE_METISRB;
        params.rtype         = METIS_RTYPE_GREEDY;
        
        params.minconn       = 0;
        params.contig        = 1;
        
        params.nooutput      = 0;
        params.wgtflag       = 3;
        
        params.ncuts         = 1;
        params.niter         = 10;
        params.ncommon       = 2;
        
        params.dbglvl        = 511;
        params.balance       = 0;
        params.seed          = -1;
        
        params.tpwgtsfile    = NULL;
        
        params.filename      = NULL;
        params.nparts        = nparts;
        
        params.ufactor       = 30;
        
        if (params.nparts < 2) {
            assert(0 && "The number of partitions should be greater than 1!\n");
        }
        
        /* Set the ptype-specific defaults */
        if (params.ptype == METIS_PTYPE_RB) {
            params.rtype = METIS_RTYPE_FM;
        }
        if (params.ptype == METIS_PTYPE_KWAY) {
            params.iptype = METIS_IPTYPE_METISRB;
            params.rtype  = METIS_RTYPE_GREEDY;
        }
        
        /* Check for invalid parameter combination */
        if (params.ptype == METIS_PTYPE_RB) {
            if (params.contig)
                assert(0 && "The -contig option cannot be specified with rb partitioning.\n");
            if (params.minconn)
                assert(0 && "The -minconn option cannot be specified with rb partitioning.\n");
            if (params.objtype == METIS_OBJTYPE_VOL)
                assert(0 && "The -objtype=vol option cannot be specified with rb partitioning.\n");
        }
    }
    
    void METIS::printInput(const idx_t *ewgt,
                           const params_t& params) const
    {
        /* The text labels for PTypes */
        static char ptypenames[][15] = {"rb", "kway"};
        /* The text labels for ObjTypes */
        static char objtypenames[][15] = {"cut", "vol", "node"};
        /* The text labels for CTypes */
        static char ctypenames[][15] = {"rm", "shem"};
        /* The text labels for RTypes */
        static char rtypenames[][15] = {"fm", "greedy", "2sided", "1sided"};
        /* The text labels for ITypes */
        static char iptypenames[][15] = {"grow", "random", "edge", "node", "metisrb"};
        /* The text labels for GTypes */
        static char gtypenames[][15] = {"dual", "nodal"};
        
        printf("Options ---------------------------------------------------------------------\n");
        printf(" ptype=%s, objtype=%s, ctype=%s, rtype=%s, iptype=%s\n",
               ptypenames[params.ptype], objtypenames[params.objtype], ctypenames[params.ctype],
               rtypenames[params.rtype], iptypenames[params.iptype]);
        printf(" dbglvl=%" PRIDX", ufactor=%.3f, minconn=%s, contig=%s, nooutput=%s\n",
               params.dbglvl,
               (1.0+0.001*(params.ufactor)),
               (params.minconn  ? "YES" : "NO"),
               (params.contig   ? "YES" : "NO"),
               (params.nooutput ? "YES" : "NO")
               );
        printf(" seed=%" PRIDX", niter=%" PRIDX", ncuts=%" PRIDX"\n",
               params.seed, params.niter, params.ncuts);
        printf(" gtype=%s, ncommon=%" PRIDX", niter=%" PRIDX", ncuts=%" PRIDX"\n",
               gtypenames[params.gtype], params.ncommon, params.niter, params.ncuts);
        printf("\n");
        switch (params.ptype) {
            case METIS_PTYPE_RB:
                printf("Recursive Partitioning ------------------------------------------------------\n");
                break;
            case METIS_PTYPE_KWAY:
                printf("Direct k-way Partitioning ---------------------------------------------------\n");
                break;
        }
        
        int i, j;
        printf("ne, nn: %ld %ld\n", ne, nn);
        for(i = 0; i < ne; i++) {
            for(j = eptr[i]; j < eptr[i+1]; j++) {
                printf("%ld ", eind[j]);
            }
            printf("\n");
        }
        printf("eptr:");
        for(i = 0; i < ne + 1; i++) {
            printf("%ld ", eptr[i]);
        }
        printf("\n");
        printf("eind:");
        for(i = 0; i < ne * 3; i++) {
            printf("%ld ", eind[i]);
        }
        printf("\n");
        
        printf("ewgt:\n");
        for(i = 0; i < ne; i++) {
            printf("%ld\n", ewgt[i]);
        }
        printf("ncommon, nparts: %ld %ld\n", params.ncommon, params.nparts);
        printf("tpwgts:\n");
        for(i = 0; i < params.nparts; i++) {
            printf("%lf ", params.tpwgts[i]);
        }
        printf("\n");
    }
    
    void METIS::printOutput(void) const
    {
        std::cout << "epart" << std::endl;
        for(int triI = 0; triI < ne; triI++) {
            std::cout << eptr[triI] << ": " <<
            eind[triI * 3] << " " << eind[triI * 3 + 1] << " " << eind[triI * 3 + 2] <<
            " - " << epart[triI] << std::endl;
        }
        std::cout << "npart" << std::endl;
        for(int triI = 0; triI < nn; triI++) {
            std::cout << npart[triI] << std::endl;
        }
    }
    
}
