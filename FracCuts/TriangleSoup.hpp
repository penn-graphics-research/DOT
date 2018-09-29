//
//  TriangleSoup.hpp
//  FracCuts
//
//  Created by Minchen Li on 8/30/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#ifndef TriangleSoup_hpp
#define TriangleSoup_hpp

#include <Eigen/Eigen>

#include <igl/massmatrix.h>

#include <set>
#include <array>

#include "Types.hpp"

namespace FracCuts{
    
    enum Primitive
    {
        P_GRID,
        P_SQUARE,
        P_SPIKES,
        P_SHARKEY,
        P_CYLINDER,
        P_INPUT
    };
    class Scaffold;
    
    // duplicate the vertices and edges of a mesh to separate its triangles,
    // adjacent triangles in the original mesh will have a cohesive edge structure to
    // indicate the connectivity
    template<int dim>
    class TriangleSoup{
    public: // owned data
        Eigen::MatrixXd V_rest; // duplicated rest vertex coordinates in 3D
        Eigen::MatrixXd V; // duplicated vertex coordinates, the dimension depends on the search space
        Eigen::MatrixXi F; // reordered triangle draw list (0, 1, 2, ...), indices based on V
        Eigen::MatrixXi cohE; // cohesive edge pairs with the 4 end vertex indices based on V
        Eigen::MatrixXi initSeams; // initial cohesive edge pairs actually
        
    public:
        const Scaffold* scaffold = NULL;
        double areaThres_AM; // for preventing degeneracy of air mesh triangles
        
    public: // owned features
        Eigen::VectorXi boundaryEdge; // 1: boundary edge, 0: interior edge
        Eigen::VectorXd edgeLen; // cohesive edge rest length, used as weights
        Eigen::SparseMatrix<double> LaplacianMtr; // 2 * V.rows() wide
        Eigen::SparseMatrix<double> massMatrix; // V.rows() wide
        double density = 1.0;
        Eigen::VectorXd triArea; // triangle rest area
        Eigen::MatrixXd triNormal;
        double surfaceArea;
        Eigen::VectorXd triAreaSq; // triangle rest squared area
        Eigen::VectorXd e0dote1; // triangle rest edge dot product
        Eigen::VectorXd e0SqLen, e1SqLen; // triangle edge rest squared length
        Eigen::VectorXd e0SqLen_div_dbAreaSq;
        Eigen::VectorXd e1SqLen_div_dbAreaSq;
        Eigen::VectorXd e0dote1_div_dbAreaSq;
        double avgEdgeLen;
        double virtualRadius;
        std::set<int> fixedVert; // for linear solve
        std::vector<bool> isFixedVert;
        Eigen::Matrix<double, 2, 3> bbox;
//        Eigen::MatrixXd cotVals; // cotangent values of rest triangle corners
        Eigen::VectorXd vertWeight; // for regional seam placement
        Eigen::VectorXd triWeight; // for weighted stencil
        std::vector<std::vector<int>> borderVerts_primitive;
        std::vector<Eigen::Matrix<double, dim, dim>> restTriInv;
        
        // indices for fast access
        std::map<std::pair<int, int>, int> edge2Tri;
        std::vector<std::set<int>> vNeighbor;
        std::map<std::pair<int, int>, int> cohEIndex;
        std::vector<std::set<std::pair<int, int>>> vFLoc;
        
        std::set<int> fracTail;
        int curFracTail;
        std::pair<int, int> curInteriorFracTails;
        double initSeamLen;
        
    public: // constructor
        // default constructor that doesn't do anything
        TriangleSoup(void);
        
        // initialize from a triangle mesh, V will be constructed from UV_mesh in 2D,
        // V_mesh will be used to initialize restShape
        TriangleSoup(const Eigen::MatrixXd& V_mesh, const Eigen::MatrixXi& F_mesh,
                     const Eigen::MatrixXd& Vt_mesh, double p_areaThres_AM = 0.0);
        
        TriangleSoup(Primitive primitive, double size = 1.0, int elemAmt = 1000);
        
    public: // API
        void computeMassMatrix(const igl::MassMatrixType type = igl::MASSMATRIX_TYPE_VORONOI);
        void computeFeatures(bool multiComp = false, bool resetFixedV = false);
        void updateFeatures(void);
        void resetFixedVert(const std::set<int>& p_fixedVert = std::set<int>());
        void addFixedVert(int vI);
        void addFixedVert(const std::vector<int>& p_fixedVert);
        
        bool checkInversion(int triI, bool mute) const;
        bool checkInversion(bool mute = false, const std::vector<int>& triangles = std::vector<int>()) const;
        
        void save(const std::string& filePath, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                  const Eigen::MatrixXd UV, const Eigen::MatrixXi& FUV = Eigen::MatrixXi()) const;
        void save(const std::string& filePath) const;
        
        void saveAsMesh(const std::string& filePath, bool scaleUV = false,
                        const Eigen::MatrixXi& SF = Eigen::MatrixXi()) const;
        
        void constructSubmesh(const Eigen::VectorXi& triangles,
                              TriangleSoup& submesh,
                              std::map<int, int>& globalVIToLocal,
                              std::map<int, int>& globalTriIToLocal) const;
        
    public: // helper function
        void computeLaplacianMtr(void);
        
        // toBound = false indicate counter-clockwise
        bool isBoundaryVert(int vI, int vI_neighbor,
                            std::vector<int>& tri_toSep, std::pair<int, int>& boundaryEdge, bool toBound = true) const;
        bool isBoundaryVert(int vI) const;
        
        void compute2DInwardNormal(int vI, Eigen::RowVector2d& normal) const;
    };
    
}

#endif /* TriangleSoup_hpp */
