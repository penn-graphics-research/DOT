//
//  OSQP.h
//  FracCuts
//
//  Created by Minchen Li on 10/15/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef OSQP_h
#define OSQP_h

#include <osqp.h>

namespace FracCuts {
    
    class OSQP
    {
        OSQPSettings *settings; // Problem settings
        OSQPData *data; // OSQPData
        OSQPWorkspace *work; // Workspace
        
    public:
        OSQP(bool printOutput = true)
        {
            settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
            // Define Solver settings as default
            osqp_set_default_settings(settings);
            settings->verbose = printOutput;
            settings->eps_abs = 0.0;
            
            data = (OSQPData *)c_malloc(sizeof(OSQPData));
            
            //TODO: update matrix using the osqp_update API to save time
            data->P = NULL;
            data->A = NULL;
            
            work = NULL;
        }
        ~OSQP(void)
        {
            if(work) {
                osqp_cleanup(work);
            }
            if(data->P) {
                c_free(data->P);
            }
            if(data->A) {
                c_free(data->A);
            }
            c_free(data);
            c_free(settings);
        }
        
        void setup(c_float *P_x, c_int P_nnz, c_int *P_i, c_int *P_p,
                   c_float *q,
                   c_float *A_x, c_int A_nnz, c_int * A_i, c_int *A_p,
                   c_float *l, c_float *u,
                   c_int n, c_int m)
        {
            if(work) {
                osqp_cleanup(work);
            }
            if(data->P) {
                c_free(data->P);
            }
            if(data->A) {
                c_free(data->A);
            }
            
            // Populate data
            data->n = n;
            data->m = m;
            data->P = csc_matrix(data->n, data->n, P_nnz, P_x, P_i, P_p);
            data->q = q;
            data->A = csc_matrix(data->m, data->n, A_nnz, A_x, A_i, A_p);
            data->l = l;
            data->u = u;
            
            // Setup workspace
            work = osqp_setup(data, settings);
        }
        
        c_float *solve(void)
        {
            // Solve Problem
            osqp_solve(work);
            // solution is in work->solution
            // primal: c_float *work->solution->x
            // dual: c_float *work->solution->y
            return work->solution->x;
        }
    };
    
}

#endif /* OSQP_h */
