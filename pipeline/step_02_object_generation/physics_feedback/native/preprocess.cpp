#include "coacd_1_0_14/model_obj.h"
#include <cstring>

namespace coacd {
bool IsManifold(Model &);
void SDFManifold(Model &, Model &, double, double);
}

// All allocations cross the C boundary through this module's matching free.
extern "C" int prepare_mesh(const double *v, int nv, const int *f, int nf,
                            int force, int resolution, double **ov, int *onv,
                            int **of, int *onf, char *error, int error_size) {
    try {
        std::vector<coacd::vec3d> points(nv);
        std::vector<coacd::vec3i> faces(nf);
        for (int i=0;i<nv;++i) for(int j=0;j<3;++j) points[i][j]=v[3*i+j];
        for (int i=0;i<nf;++i) for(int j=0;j<3;++j) faces[i][j]=f[3*i+j];
        coacd::Model m;
        m.Load(points, faces);
        auto bounds=m.Normalize();
        bool rebuild=force || !coacd::IsManifold(m);
        if (rebuild) {
            coacd::Model result;
            coacd::SDFManifold(m, result, resolution, 0.55);
            m.points.swap(result.points);
            m.triangles.swap(result.triangles);
            m.Recover(bounds);
        } else {
            // Keep input coordinates exactly; the check may correct winding.
            m.points=points;
        }
        *onv=static_cast<int>(m.points.size());
        *onf=static_cast<int>(m.triangles.size());
        *ov=new double[3*(*onv)];
        *of=new int[3*(*onf)];
        for(int i=0;i<*onv;++i) for(int j=0;j<3;++j) (*ov)[3*i+j]=m.points[i][j];
        for(int i=0;i<*onf;++i) for(int j=0;j<3;++j) (*of)[3*i+j]=m.triangles[i][j];
        return rebuild ? 1 : 0;
    } catch (const std::exception &e) {
        std::snprintf(error,error_size,"%s",e.what());
        return -1;
    }
}
extern "C" void release_mesh(double *v, int *f) { delete[] v; delete[] f; }
