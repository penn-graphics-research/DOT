#ifndef get_feasible_steps_hpp
#define get_feasible_steps_hpp

#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

double getSmallestPositiveRealQuadRoot(double a, double b, double c,
                                       double tol);


void computeInjectiveStepSize_2d(const MatrixXi& F,
                                 const MatrixXd& x,
                                 const VectorXd& p,
                                 double tol,
                                 double *output);


double getSmallestPositiveRealCubicRoot(double a, double b, double c, double d,
                                        double tol);


void computeInjectiveStepSize_3d(const MatrixXi& F,
                                 const MatrixXd& x,
                                 const VectorXd& p,
                                 double tol,
                                 double *output);

#endif
