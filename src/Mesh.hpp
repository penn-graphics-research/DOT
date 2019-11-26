//
//  Mesh.hpp
//  DOT
//
//  Created by Minchen Li on 8/30/17.
//

#ifndef Mesh_hpp
#define Mesh_hpp

#include <Eigen/Eigen>

#include <igl/massmatrix.h>

#include <set>
#include <array>

#include "Types.hpp"

namespace DOT{
    
    enum Primitive
    {
        P_GRID,
        P_SQUARE,
        P_RECTANGLE,
        P_SPIKES,
        P_SHARKEY,
        P_CYLINDER,
        P_INPUT
    };
    
    // duplicate the vertices and edges of a mesh to separate its triangles,
    // adjacent triangles in the original mesh will have a cohesive edge structure to
    // indicate the connectivity
    template<int dim>
    class Mesh{
    public: // owned data
        Eigen::MatrixXd V_rest; // duplicated rest vertex coordinates in 3D
        Eigen::MatrixXd V; // duplicated vertex coordinates, the dimension depends on the search space
        Eigen::MatrixXi F; // reordered triangle draw list (0, 1, 2, ...), indices based on V
        
    public: // owned features
        Eigen::SparseMatrix<double> massMatrix; // V.rows() wide
        double density, m_YM, m_PR;
        Eigen::VectorXd u, lambda;
        Eigen::VectorXd triArea; // triangle rest area
        double surfaceArea;
        double avgEdgeLen;
        std::set<int> fixedVert; // for linear solve
        std::vector<bool> isFixedVert;
        Eigen::Matrix<double, 2, 3> bbox;
        Eigen::VectorXd triWeight; // for weighted stencil
        std::vector<std::vector<int>> borderVerts_primitive;
        std::vector<Eigen::Matrix<double, dim, dim>> restTriInv;
        std::vector<bool> m_isBoundaryVert;
        
        // indices for fast access
        std::vector<std::set<int>> vNeighbor;
        std::vector<std::set<std::pair<int, int>>> vFLoc;
        
    public: // constructor
        // default constructor that doesn't do anything
        Mesh(void);
        
        // initialize from a triangle mesh, V will be constructed from UV_mesh in 2D,
        // V_mesh will be used to initialize restShape
        Mesh(const Eigen::MatrixXd& V_mesh, const Eigen::MatrixXi& F_mesh,
                     const Eigen::MatrixXd& Vt_mesh,
                     double YM, double PR, double rho);
        
        Mesh(Primitive primitive, double size, int elemAmt,
                     double YM, double PR, double rho);
        
    public: // API
        void computeMassMatrix(const igl::MassMatrixType type = igl::MASSMATRIX_TYPE_VORONOI);
        void computeFeatures(bool multiComp = false, bool resetFixedV = false);
        void resetFixedVert(const std::set<int>& p_fixedVert = std::set<int>());
        void addFixedVert(int vI);
        void addFixedVert(const std::vector<int>& p_fixedVert);
        void removeFixedVert(int vI);
        
        void setLameParam(double YM, double PR);
        
        bool checkInversion(int triI, bool mute) const;
        bool checkInversion(bool mute = false, const std::vector<int>& triangles = std::vector<int>()) const;
        
        void saveAsMesh(const std::string& filePath, bool scaleUV = false,
                        const Eigen::MatrixXi& SF = Eigen::MatrixXi()) const;
        
        void constructSubmesh(const Eigen::VectorXi& triangles,
                              Mesh& submesh,
                              std::map<int, int>& globalVIToLocal,
                              std::map<int, int>& globalTriIToLocal) const;
        
    public: // helper function
        bool isBoundaryVert(int vI) const;
        void computeBoundaryVert(const Eigen::MatrixXi& SF);
    };
    
}

#endif /* Mesh_hpp */
