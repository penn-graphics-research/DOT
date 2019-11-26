//
//  METIS.hpp
//  DOT
//
//  Created by Minchen Li on 8/3/18.
//

#ifndef METIS_hpp
#define METIS_hpp

#include "Mesh.hpp"

#include <metis.h>

namespace DOT {
    
    // data structure from METIS mesh partitioning example
    typedef struct {
        idx_t ptype;    // partitioning method:
                        // METIS_PTYPE_RB, METIS_PTYPE_KWAY
        
        idx_t objtype;  // objective:
                        // METIS_OBJTYPE_CUT, METIS_OBJTYPE_VOL
        
        idx_t ctype;    // matching scheme for coarsening:
                        // METIS_CTYPE_RM, METIS_CTYPE_SHEM
        
        idx_t iptype;   // algorithm for initial partitioning:
                        // METIS_IPTYPE_GROW, METIS_IPTYPE_RANDOM,
                        // METIS_IPTYPE_EDGE, METIS_IPTYPE_NODE
        
        idx_t rtype;    // algorithm for refinement
                        // METIS_RTYPE_FM, METIS_RTYPE_GREEDY,
                        //  METIS_RTYPE_SEP2SIDED, METIS_RTYPE_SEP1SIDED
        
        idx_t no2hop;   // perform a 2-hop matching (0) or not (1) for coarsening
        idx_t minconn;  // whether to minimize the degree of the subdomain graph (1) or not (0)
        idx_t contig;   // whether partition needs to be continuous (1) or not (0)
        
        idx_t nooutput;
        
        idx_t balance;
        idx_t ncuts;    // number of candidate partition plans
        idx_t niter;    // number of iterations for refinement
        
        idx_t gtype;    // type of graph (dual or primal)
        idx_t ncommon;  // number of common nodes to have for elements to be considered adjacent
        
        idx_t seed;     // seed for random number generation
        idx_t dbglvl;   // amount of debug info to be printed
        
        idx_t nparts;   // number of partitionings
        
        idx_t nseps;    // number of different candidate separators
        idx_t ufactor;  // allowed load imbalance (1+ufactor)/1000
        idx_t pfactor;  // if pfactor > 0, vertices with a degree greater than 0.1*pfactor*avgDeg
                        // will be removed during ordering and then add back. 60 <= x <= 200 is
                        // often good
        
        idx_t compress; // compress redundant vertices (1) or not (0)
        idx_t ccorder;  // whether to order connected components (1) or not (0)
        
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
    
    template<int dim>
    class METIS
    {
    protected: // data
        idx_t nn, ne; // METIS mesh info
        std::vector<idx_t> eptr, eind; // METIS mesh data structure
        std::vector<idx_t> epart, npart; // output partition info
        
    public: // constructor
        METIS(const Mesh<dim>& mesh)
        {
            // construct METIS mesh representation
            eptr.resize(mesh.F.rows() + 1);
            eind.resize(mesh.F.rows() * (dim + 1));
            for(int triI = 0; triI < mesh.F.rows(); triI++) {
                eptr[triI] = triI * (dim + 1);
                const Eigen::Matrix<int, 1, (dim + 1)>& triVInd = mesh.F.row(triI);
                for(int i = 0; i < (dim + 1); i++) {
                    eind[eptr[triI] + i] = triVInd[i];
                }
            }
            eptr[mesh.F.rows()] = mesh.F.rows() * (dim + 1);
            
            ne = mesh.F.rows();
            nn = mesh.V_rest.rows();
        }
        
    public: // API
        void partMesh(idx_t nparts, idx_t *p_ewgt = NULL)
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
            options[METIS_OPTION_MINCONN] = params.minconn;
            options[METIS_OPTION_CONTIG]  = params.contig;
            options[METIS_OPTION_NCUTS]   = params.ncuts;
            options[METIS_OPTION_NSEPS]   = params.nseps;
            options[METIS_OPTION_NITER]   = params.niter;
            options[METIS_OPTION_DBGLVL]  = params.dbglvl;
            options[METIS_OPTION_SEED]    = params.seed;
            options[METIS_OPTION_UFACTOR] = params.ufactor;
            
            // init other parameters
            epart.resize(ne);
            npart.resize(nn);
            std::vector<idx_t> ewgt(ne, 1); // element weights, computational cost
            std::vector<real_t> tpwgts(nparts, 1.0 / nparts); // part weights
            idx_t objval;
            
            int status = METIS_PartMeshDual(&ne, &nn, eptr.data(), eind.data(),
                                            p_ewgt ? p_ewgt : ewgt.data(), NULL, // communication cost
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
        void partMesh_nodes(idx_t nparts, idx_t *p_vwgt = NULL)
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
            options[METIS_OPTION_MINCONN] = params.minconn;
            options[METIS_OPTION_CONTIG]  = params.contig;
            options[METIS_OPTION_NCUTS]   = params.ncuts;
            options[METIS_OPTION_NSEPS]   = params.nseps;
            options[METIS_OPTION_NITER]   = params.niter;
            options[METIS_OPTION_DBGLVL]  = params.dbglvl;
            options[METIS_OPTION_SEED]    = params.seed;
            options[METIS_OPTION_UFACTOR] = params.ufactor;
            
            // init other parameters
            epart.resize(ne);
            npart.resize(nn);
            std::vector<idx_t> vwgt(nn, 1); // element weights, computational cost
            std::vector<real_t> tpwgts(nparts, 1.0 / nparts); // part weights
            idx_t objval;
            
            int status = METIS_PartMeshNodal(&ne, &nn, eptr.data(), eind.data(),
                                             p_vwgt ? p_vwgt : vwgt.data(), NULL,
                                             &params.nparts, tpwgts.data(), options,
                                             &objval, epart.data(), npart.data());
            
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
        void partMesh_slice(const Mesh<dim>& mesh,
                            int nParts, int dimI)
        {
            Eigen::VectorXd center_dimI(mesh.F.rows());
            center_dimI.setZero();
            for(int elemI = 0; elemI < mesh.F.rows(); elemI++) {
                const Eigen::Matrix<int, 1, dim + 1>& elemVInd = mesh.F.row(elemI);
                for(int vI = 0; vI < dim + 1; vI++) {
                    center_dimI[elemI] += mesh.V(elemVInd[vI], dimI);
                }
                center_dimI[elemI] /= dim + 1;
            }
            
            epart.resize(mesh.F.rows());
            double step = (mesh.V.col(dimI).maxCoeff() - mesh.V.col(dimI).minCoeff()) / nParts;
            for(int elemI = 0; elemI < mesh.F.rows(); elemI++) {
                epart[elemI] = std::min(nParts - 1, std::max(0, int(center_dimI[elemI] / step)));
            }
        }
        void getElementList(int partI, Eigen::VectorXi& elementListI) const
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
        void getNodeList(int partI, Eigen::VectorXi& nodeListI) const
        {
            assert(!npart.empty());
            
            // load element list
            nodeListI.resize(0);
            for(int vI = 0; vI < npart.size(); ++vI) {
                if(npart[vI] == partI) {
                    int oldSize = nodeListI.size();
                    nodeListI.conservativeResize(oldSize + 1);
                    nodeListI[oldSize] = vI;
                }
            }
        }
        const std::vector<idx_t>& getNpart(void) const {
            return npart;
        }
        
    protected: // helper
        void initParam(params_t& params, idx_t nparts)
        {
            memset((void *)&params, 0, sizeof(params_t));
            
            /* initialize the params data structure */
            params.gtype         = METIS_GTYPE_DUAL;
            params.ptype         = METIS_PTYPE_KWAY;
            params.objtype       = METIS_OBJTYPE_CUT;
            params.ctype         = METIS_CTYPE_SHEM;
            params.iptype        = METIS_IPTYPE_METISRB;
            params.rtype         = METIS_RTYPE_GREEDY;
            
            params.minconn       = 1;
            params.contig        = 1;
            
            params.nooutput      = 0;
            params.wgtflag       = 3;
            
            params.ncuts         = 3; // originally 1
            params.nseps         = 3; // originally 1
            params.niter         = 10;
            params.ncommon       = dim;
            
            params.dbglvl        = 511;
            params.balance       = 0;
            params.seed          = -1;
            
            params.tpwgtsfile    = NULL;
            
            params.filename      = NULL;
            params.nparts        = nparts;
            
            params.ufactor       = 30; //
            
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
        
        void printInput(const idx_t *ewgt,
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
        void printOutput(void) const
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
    };
    
}

#endif /* METIS_hpp */
