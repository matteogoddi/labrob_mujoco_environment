#pragma once
#include <Eigen/Dense>
#include <cmath>

constexpr int NX = 9;
constexpr int NU = 3;
constexpr int NY = 3;
constexpr int NC = 2 * NY;
constexpr int NH = 20;

// pendubot parameters
static const double h = 0.75;
static const double grav = 9.81;
static const double η = sqrt(grav/h);
static const double Δ = 0.01;

// cost function weights
static const double w_z  = 100.0;
static const double w_zd = 1.0;
static const double w_term = 0; //1e7;

// reference ZMP
Eigen::Matrix<double, NY, NX> C_zmp =
(Eigen::Matrix<double, NY, NX>() << 0, 0, 1, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 1, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 1).finished();
Eigen::Matrix<double, NY, NX> C_com =
(Eigen::Matrix<double, NY, NX>() << 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 1, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 1, 0, 0).finished();
Eigen::Matrix<double, NY, NH+1> zmp_ref = Eigen::Matrix<double, NY, NH+1>::Zero();

// dynamics
Eigen::Matrix<double, 3, 3> A_LIP = (Eigen::Matrix<double, 3, 3>() << 0, 1, 0,
                                                                η*η, 0, -η*η,
                                                                0, 0, 0).finished(); 

Eigen::Matrix<double, 3, 1> B_LIP = (Eigen::Matrix<double, 3, 1>() << 0, 0, 1).finished();

Eigen::Matrix<double, NX, NX> I = Eigen::Matrix<double, NX, NX>::Identity();

Eigen::Matrix<double, NX, NX> A = Eigen::Matrix<double, NX, NX>::Zero();
Eigen::Matrix<double, NX, NU> B = Eigen::Matrix<double, NX, NU>::Zero();
inline void initialize_problem()
{
    A.block<3,3>(0,0) = A_LIP;
    A.block<3,3>(3,3) = A_LIP;
    A.block<3,3>(6,6) = A_LIP;
    B.block<3,1>(0,0) = B_LIP;
    B.block<3,1>(3,1) = B_LIP;
    B.block<3,1>(6,2) = B_LIP;
}

inline void compute_ref()
{
    double step_x = 0.1;
    double step_y = 0.1;
    for (int i = 0; i < NH+1; ++i)
    {
        int step_index = floor(i / 100);
        int left_right = step_index % 2 == 0 ? 1 : -1;
        zmp_ref(0,i) = step_x * step_index;
        zmp_ref(1,i) = step_y * left_right;
        zmp_ref(2,i) = 0.0;
    }
}

// constraints
Eigen::Matrix<double, NC, 1> g(const Eigen::Matrix<double, NX, 1> &x, const int &i)
{
    Eigen::Matrix<double, NY, 1> zmp = C_zmp * x;
    Eigen::Matrix<double, NY, 1> zmp_error = zmp - zmp_ref.col(i);

    Eigen::Matrix<double, NC, 1> g;
    double constraint_half_width = 0.05;
    
    // upper boundary
    g(0,0) = - (constraint_half_width - zmp_error(0,0));
    g(1,0) = - (constraint_half_width - zmp_error(1,0));
    g(2,0) = - (constraint_half_width - zmp_error(2,0));

    // lower boundary
    g(3,0) = - (zmp_error(0,0) + constraint_half_width);
    g(4,0) = - (zmp_error(1,0) + constraint_half_width);
    g(5,0) = - (zmp_error(2,0) + constraint_half_width);

    return g;
}

Eigen::Matrix<double, NC, NX> gx(const Eigen::Matrix<double, NX, 1> &x, const int &i)
{
    Eigen::Matrix<double, NC, NX> gx;
    gx << C_zmp, -C_zmp;
    return gx;
}

Eigen::Matrix<double, NC, NU> gu(const Eigen::Matrix<double, NX, 1> &x, const int &i)
{
    return Eigen::Matrix<double, NC, NU>::Zero();
}

// dynamics
inline Eigen::Matrix<double, NX, 1> f(const Eigen::Matrix<double, NX, 1> &x, const Eigen::Matrix<double, NU, 1> &u)
{
    return x + Δ * (A * x + B * u);
}

inline Eigen::Matrix<double, NX, NX> fx(const Eigen::Matrix<double, NX, 1> &x, const Eigen::Matrix<double, NU, 1> &u)
{
    return I + Δ * A;
}

inline Eigen::Matrix<double, NX, NU> fu(const Eigen::Matrix<double, NX, 1> &x, const Eigen::Matrix<double, NU, 1> &u)
{
    return Δ * B;
}

// terminal constraint
inline Eigen::Matrix<double, NY, 1> h_ter(const Eigen::Matrix<double, NX, 1> &x)
{
    return (C_com - C_zmp) * x;
}

inline Eigen::Matrix<double, NY, NX> h_terx(const Eigen::Matrix<double, NX, 1> &x)
{
    return C_com - C_zmp;
}

// cost terms
inline Eigen::Matrix<double, 1, 1> L(const Eigen::Matrix<double, NX, 1> &x, const Eigen::Matrix<double, NU, 1> &u, const int &i)
{
    Eigen::Matrix<double, NY, 1> zmp = C_zmp * x;
    Eigen::Matrix<double, NY, 1> zmp_error = zmp - zmp_ref.col(i);
    Eigen::Matrix<double, 1, 1> cost;
    cost(0,0) = w_z * zmp_error.squaredNorm() + w_zd * u.squaredNorm();
    return cost;
}

inline Eigen::Matrix<double, NX, 1> Lx(const Eigen::Matrix<double, NX, 1> &x, const Eigen::Matrix<double, NU, 1> &u, const int &i)
{
    Eigen::Matrix<double, NY, 1> zmp = C_zmp * x;
    Eigen::Matrix<double, NY, 1> zmp_error = zmp - zmp_ref.col(i);
    return 2 * w_z * C_zmp.transpose() * zmp_error;
}

inline Eigen::Matrix<double, NU, 1> Lu(const Eigen::Matrix<double, NX, 1> &x, const Eigen::Matrix<double, NU, 1> &u, const int &i)
{
    return 2 * w_zd * u;
}

inline Eigen::Matrix<double, NX, NX> Lxx(const Eigen::Matrix<double, NX, 1> &x, const Eigen::Matrix<double, NU, 1> &u, const int &i)
{
    Eigen::Matrix<double, NY, 1> zmp = C_zmp * x;
    Eigen::Matrix<double, NY, 1> zmp_error = zmp - zmp_ref.col(i);
    return 2 * w_z * C_zmp.transpose() * C_zmp;
}

inline Eigen::Matrix<double, NU, NU> Luu(const Eigen::Matrix<double, NX, 1> &x, const Eigen::Matrix<double, NU, 1> &u, const int &i)
{
    return 2 * w_zd * Eigen::Matrix<double, NU, NU>::Identity();
}

inline Eigen::Matrix<double, NU, NX> Lux(const Eigen::Matrix<double, NX, 1> &x, const Eigen::Matrix<double, NU, 1> &u, const int &i)
{
    return Eigen::Matrix<double, NU, NX>::Zero();
}

inline Eigen::Matrix<double, 1, 1> L_ter(const Eigen::Matrix<double, NX, 1> &x)
{
    Eigen::Matrix<double, 1, 1> cost;
    cost(0,0) = w_term * ((C_com - C_zmp) * x).squaredNorm();
    return cost;
    //return Eigen::Matrix<double, 1, 1>::Zero();
}

inline Eigen::Matrix<double, NX, 1> L_terx(const Eigen::Matrix<double, NX, 1> &x)
{
    return 2 * w_term * (C_com - C_zmp).transpose() * (C_com - C_zmp) * x;
    //return Eigen::Matrix<double, NX, 1>::Zero();
}

inline Eigen::Matrix<double, NX, NX> L_terxx(const Eigen::Matrix<double, NX, 1> &x)
{
    return 2 * w_term * (C_com - C_zmp).transpose() * (C_com - C_zmp);
    //return Eigen::Matrix<double, NX, NX>::Zero();
}
