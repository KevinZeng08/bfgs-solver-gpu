﻿// https://aria42.com/blog/2014/12/understanding-lbfgs
#define _CRT_SECURE_NO_WARNINGS
#define USE_LBFGS

#include <assert.h>
#include <deque>
#include <iostream>
#include <map>
#include <math.h>
#include <omp.h>
#include <string>
#include <vector>

#include "array2.h"
#include "ops.h"

#define START_CPU                                                              \
  {                                                                            \
    double start = omp_get_wtime();

#define END_CPU(name)                                                          \
  double end = omp_get_wtime();                                                \
  double duration = end - start;                                               \
  printf("%s Time used: %3.1f ms\n", name, duration * 1000);                   \
  }

std::vector<int> objEqHeads, gradEqHeads;  // start offset
std::vector<double> objEqVals, gradEqVals; // results for equations

enum NodeType { NODE_CONST, NODE_OPER, NODE_VAR };

enum OpType {
  OP_PLUS = 0,
  OP_MINUS = 1,
  OP_UMINUS = 2,
  OP_TIME = 3,
  OP_DIVIDE = 4,
  OP_SIN,
  OP_COS,
  OP_TG,
  OP_CTG,
  OP_SEC,
  OP_CSC,
  OP_ARCSIN,
  OP_ARCCOS,
  OP_ARCTG,
  OP_ARCCTG,
  OP_POW,
  OP_EXP,
  OP_EEXP,
  OP_SQR,
  OP_SQRT,
  OP_LOG,
  OP_LN,
  OP_NULL = -1
};

typedef struct _EqInfo {
  NodeType _type;
  double _val;
  int _var;
  OpType _op;
  int _left;
  int _right;
} EqInfo;

enum VarType { VAR_CONST, VAR_UNSOLVED, VAR_SOLVED, VAR_DELETED, VAR_FREE };

enum OptimType { BFGS, LBFGS };

struct VarInfo {
  VarType _type;
  double _val;

  VarInfo(VarType ty, double val) : _type(ty), _val(val) {}
};

double epsZero1 = 1e-20;
double epsZero2 = 1e-7;
#ifndef M_PI_2
#define M_PI_2 (1.57079632679489661923)
#endif

#define BFGS_MAXIT 500
#define BFGS_STEP 0.1

static int _GetMaxIt() { return BFGS_MAXIT; }

static double _GetStep() { return BFGS_STEP; }

static double _GetEps() { return 0.01; }

static void _ConstructVarTab(std::vector<double> &vars,
                             std::vector<int> &varMap,
                             std::vector<int> &revMap);
static void _ConstructObjEqTab(std::vector<EqInfo> &eqs, int &numEqs,
                               const std::vector<int> &revMap);
static void _ConstructGradEqTab(std::vector<EqInfo> &eqs, int &numEqs,
                                const std::vector<int> &revMap);
static void _ScatterVarTab(std::vector<double> &x, std::vector<int> &varMap);

static void _VecCopy(std::vector<double> &dst, const std::vector<double> &src) {
  int n = src.size();
  for (int i = 0; i < n; i++)
    dst[i] = src[i];
}

static void _VecSub(const std::vector<double> &a, const std::vector<double> &b,
                    std::vector<double> &ret) {
  int n = a.size();
  for (int i = 0; i < n; i++)
    ret[i] = a[i] - b[i];
}

static double _VecDot(const std::vector<double> &a,
                      const std::vector<double> &b) {
  double s = 0;
  int n = a.size();
  for (int i = 0; i < n; i++)
    s += a[i] * b[i];

  return s;
}

static void _VecMult(std::vector<double> &v, double t) {
  int n = v.size();
  for (int i = 0; i < n; i++)
    v[i] *= t;
}

static double _VecAdd(const std::vector<double> &a,
                      const std::vector<double> &b, std::vector<double> &ret) {
  int n = a.size();
  for (int i = 0; i < n; i++)
    ret[i] = a[i] + b[i];
}

static void _VecAxPy(const std::vector<double> &a, double t,
                     const std::vector<double> &b, std::vector<double> &ret) {
  int n = a.size();
  for (int i = 0; i < n; i++)
    ret[i] = a[i] + b[i] * t;
}

static double _VecLen(const std::vector<double> &v) {
  return sqrt(_VecDot(v, v));
}

static void _VecNorm(std::vector<double> &v) {
  double tmp = _VecLen(v);
  if (tmp > 0.0) {
    _VecMult(v, 1.0 / tmp);
  }
}

static void _CalcEqNew2(const std::vector<double> &x,
                        const std::vector<EqInfo> &etab, int st, int ed,
                        std::vector<double> &vtab) {
  for (int i = ed - 1; i >= st; i--) {
    const EqInfo &eq = etab[i];
    switch (eq._type) {
    case NODE_CONST:
      vtab[i] = eq._val;
      break;

    case NODE_VAR: {
      int idx = eq._var;
      vtab[i] = x[idx];
      break;
    }

    case NODE_OPER: {
      double left = vtab[eq._left];
      double right = vtab[eq._right];
      switch (eq._op) {
      case OP_PLUS:
        vtab[i] = (left + right);
        break;
      case OP_MINUS:
        vtab[i] = (left - right);
        break;
      case OP_UMINUS:
        vtab[i] = -right;
        break;
      case OP_TIME:
        vtab[i] = (left * right);
        break;
      case OP_DIVIDE:
        vtab[i] = (left / right);
        break;
      case OP_SIN:
        vtab[i] = (sin(left));
        break;
      case OP_COS:
        vtab[i] = (cos(left));
        break;
      case OP_TG:
        vtab[i] = (tan(left));
        break;
      case OP_CTG:
        vtab[i] = (1.0 / tan(left));
        break;
      case OP_SEC:
        vtab[i] = (1.0 / cos(left));
        break;
      case OP_CSC:
        vtab[i] = (1.0 / sin(left));
        break;
      case OP_ARCSIN:
        vtab[i] = (asin(left));
        break;
      case OP_ARCCOS:
        vtab[i] = (acos(left));
        break;
      case OP_ARCTG:
        vtab[i] = (atan(left));
        break;
      case OP_ARCCTG:
        vtab[i] = (atan(-left) + M_PI_2);
        break;
      case OP_POW:
        vtab[i] = (pow(left, right));
        break;
      case OP_EEXP:
        vtab[i] = (exp(left));
        break;
      case OP_EXP:
        vtab[i] = (exp(left * log(right)));
        break;
      case OP_LN:
        vtab[i] = (log(left));
        break;
      case OP_LOG:
        vtab[i] = (log(right) / log(left));
        break;
      case OP_SQR:
        vtab[i] = (left * left);
        break;
      case OP_SQRT:
        vtab[i] = (sqrt(left));
        break;
      default:
        fprintf(stderr, "Unknown operator in EsCalcTree()\n");
        assert(0);
      }
    }
    }
  }
}

static double _CalcEqNew1(const std::vector<double> &x, const EqInfo &eq,
                          const std::vector<EqInfo> &etab, int item,
                          const std::vector<int> &htab, int allNum,
                          std::vector<double> &vtab) {
  int ed = item < 0 ? allNum : htab[item + 1];
  int st = item < 0 ? htab[-item] : htab[item];
  _CalcEqNew2(x, etab, st, ed, vtab); // vtab: values for equation units

  switch (eq._type) {
  case NODE_OPER: {
    double left = vtab[eq._left];
    double right = vtab[eq._right];
    switch (eq._op) {
    case OP_PLUS:
      return (left + right);
    case OP_MINUS:
      return (left - right);
    case OP_UMINUS:
      return (-right);
    case OP_TIME:
      return (left * right);
    case OP_DIVIDE:
      return (left / right);
    case OP_SIN:
      return (sin(left));
    case OP_COS:
      return (cos(left));
    case OP_TG:
      return (tan(left));
    case OP_CTG:
      return (1.0 / tan(left));
    case OP_SEC:
      return (1.0 / cos(left));
    case OP_CSC:
      return (1.0 / sin(left));
    case OP_ARCSIN:
      return (asin(left));
    case OP_ARCCOS:
      return (acos(left));
    case OP_ARCTG:
      return (atan(left));
    case OP_ARCCTG:
      return (atan(-left) + M_PI_2);
    case OP_POW:
      return (pow(left, right));
    case OP_EEXP:
      return (exp(left));
    case OP_EXP:
      return (exp(left * log(right)));
    case OP_LN:
      return (log(left));
    case OP_LOG:
      return (log(right) / log(left));
    case OP_SQR:
      return (left * left);
    case OP_SQRT:
      return (sqrt(left));
    default:
      fprintf(stderr, "Unknown operator in EsCalcTree()\n");
      assert(0);
      return (0.0);
    }
  }
  }

  assert(0);
  return 0;
}

static double _CalcEq(const std::vector<double> &x, const EqInfo &eq,
                      const std::vector<EqInfo> &etab) {
  double left, right;

  switch (eq._type) {
  case NODE_CONST:
    return (eq._val);
    break;

  case NODE_VAR: {
    int idx = eq._var;
    return x[idx];
    break;
  }

  case NODE_OPER: {
    left = _CalcEq(x, etab[eq._left], etab);
    right = _CalcEq(x, etab[eq._right], etab);
    switch (eq._op) {
    case OP_PLUS:
      return (left + right);
    case OP_MINUS:
      return (left - right);
    case OP_UMINUS:
      return (-right);
    case OP_TIME:
      return (left * right);
    case OP_DIVIDE:
      return (left / right);
    case OP_SIN:
      return (sin(left));
    case OP_COS:
      return (cos(left));
    case OP_TG:
      return (tan(left));
    case OP_CTG:
      return (1.0 / tan(left));
    case OP_SEC:
      return (1.0 / cos(left));
    case OP_CSC:
      return (1.0 / sin(left));
    case OP_ARCSIN:
      return (asin(left));
    case OP_ARCCOS:
      return (acos(left));
    case OP_ARCTG:
      return (atan(left));
    case OP_ARCCTG:
      return (atan(-left) + M_PI_2);
    case OP_POW:
      return (pow(left, right));
    case OP_EEXP:
      return (exp(left));
    case OP_EXP:
      return (exp(left * log(right)));
    case OP_LN:
      return (log(left));
    case OP_LOG:
      return (log(right) / log(left));
    case OP_SQR:
      return (left * left);
    case OP_SQRT:
      return (sqrt(left));
    default:
      fprintf(stderr, "Unknown operator in EsCalcTree()\n");
      assert(0);
      return (0.0);
    }
  }
  }

  assert(0);
  return 0;
}

static double _CalcObj(const std::vector<double> &x,
                       const std::vector<EqInfo> &eqs, int eqNum) {
  std::vector<double> tmp;
  tmp.resize(eqNum);

  for (int i = 0; i < eqNum; ++i) {
    assert(eqs[i]._op == OP_PLUS || eqs[i]._op == OP_MINUS);
  }
  for (int i = 0; i < eqNum; i++) {
    // double v1 = _CalcEq(x, eqs[i], eqs);
    double v2 = _CalcEqNew1(x, eqs[i], eqs, i == eqNum - 1 ? -i : i, objEqHeads,
                            objEqVals.size(), objEqVals);
    // assert(v1 == v2);
    tmp[i] = v2;
  }

  return _VecDot(tmp, tmp);
}

static void _CalcGrad(const std::vector<double> &x, std::vector<double> &g,
                      const std::vector<EqInfo> &eqs) {
  int n = x.size();
  for (int i = 0; i < n; i++) {
    // double v1 = _CalcEq(x, eqs[i], eqs);
    double v2 = _CalcEqNew1(x, eqs[i], eqs, i == n - 1 ? -i : i, gradEqHeads,
                            gradEqVals.size(), gradEqVals);
    // assert(v1 == v2);
    g[i] = v2;
  }
}

static double _CalcObj(const std::vector<double> &x0, double h,
                       const std::vector<double> &p,
                       const std::vector<EqInfo> &eqs, int eqNum) {
  std::vector<double> xt;
  xt.resize(x0.size());
  _VecAxPy(x0, h, p, xt);
  return _CalcObj(xt, eqs, eqNum);
}

static void _CalcyTH(const std::vector<double> &y, const array2<double> &H,
                     std::vector<double> &yTH) {
  int i, j;
  int n = y.size();

  _GEMVCpu<double>(H.data(), CPULayout::COL_MAJOR, y.data(), yTH.data(),
                   H.rows(), H.cols());
}

static void _CalcHy(const array2<double> &H, const std::vector<double> &y,
                    std::vector<double> &Hy) {
  _GEMVCpu<double>(H.data(), CPULayout::ROW_MAJOR, y.data(), Hy.data(),
                   H.rows(), H.cols());
}

static void _Calcp(const array2<double> &H, const std::vector<double> &g,
                   std::vector<double> &p) {
  _CalcHy(H, g, p);

  int n = p.size();
  while (n--)
    p[n] = -p[n];
}

static void _UpdateH(array2<double> &H, const std::vector<double> &yTH,
                     const std::vector<double> &y, double sy,
                     const std::vector<double> &s,
                     const std::vector<double> &Hy) {
  int n = y.size();
  _UpdateHCpu(H.data(), y.data(), yTH.data(), sy, s.data(), Hy.data(), n);
}

#define BFGS_MAXBOUND 1e+10
static void _DetermineInterval(const std::vector<double> &x0, double h,
                               const std::vector<double> &p, double *left,
                               double *right, const std::vector<EqInfo> &eqs,
                               int eqNum) {
  double A, B, C, D, u, v, w, s, r;

  A = _CalcObj(x0, 0.0, p, eqs, eqNum);
  B = _CalcObj(x0, h, p, eqs, eqNum);
  if (B > A) {
    s = -h;
    C = _CalcObj(x0, s, p, eqs, eqNum);
    if (C > A) {
      *left = -h;
      *right = h;
      return;
    }
    B = C;
  } else {
    s = h;
  }
  u = 0.0;
  v = s;
  while (1) {
    s += s;
    if (fabs(s) > BFGS_MAXBOUND) {
      *left = *right = 0.0;
      return;
    }
    w = v + s;
    C = _CalcObj(x0, w, p, eqs, eqNum);
    if (C >= B)
      break;
    u = v;
    A = B;
    v = w;
    B = C;
  }
  r = (v + w) * 0.5;
  D = _CalcObj(x0, r, p, eqs, eqNum);
  if (s < 0.0) {
    if (D < B) {
      *left = w;
      *right = v;
    } else {
      *left = r;
      *right = u;
    }
  } else {
    if (D < B) {
      *left = v;
      *right = w;
    } else {
      *left = u;
      *right = r;
    }
  }
}

static void _GodenSep(const std::vector<double> &x0,
                      const std::vector<double> &p, double left, double right,
                      std::vector<double> &x, const std::vector<EqInfo> &eqs,
                      int eqNum) {
  static double beta = 0.61803398874989484820;
  double t1, t2, f1, f2;

  t2 = left + beta * (right - left);
  f2 = _CalcObj(x0, t2, p, eqs, eqNum);
ENTRY1:
  t1 = left + right - t2;
  f1 = _CalcObj(x0, t1, p, eqs, eqNum);
ENTRY2:
  if (fabs(t1 - t2) < epsZero2) {
    t1 = (t1 + t2) / 2.0;
    // printf("LineSearch t = %lf\n", t1*10000);

    _VecAxPy(x0, t1, p, x);
    return;
  }
  if ((fabs(left) > BFGS_MAXBOUND) || (fabs(left) > BFGS_MAXBOUND))
    return;
  if (f1 <= f2) {
    right = t2;
    t2 = t1;
    f2 = f1;
    goto ENTRY1;
  } else {
    left = t1;
    t1 = t2;
    f1 = f2;
    t2 = left + beta * (right - left);
    f2 = _CalcObj(x0, t2, p, eqs, eqNum);
    goto ENTRY2;
  }
}

static void _LinearSearch(const std::vector<double> &x0,
                          const std::vector<double> &p, double h,
                          std::vector<double> &x,
                          const std::vector<EqInfo> &eqs, int eqNum) {
  double left, right;

  _DetermineInterval(x0, h, p, &left, &right, eqs, eqNum);
  if (left == right)
    return;

  // printf("%lf, %lf\n", left, right);
  _GodenSep(x0, p, left, right, x, eqs, eqNum);
}

#define H_EPS1 1e-5
#define H_EPS2 1e-5
#define H_EPS3 1e-4

static bool _HTerminate(const std::vector<double> &xPrev,
                        const std::vector<double> &xNow, double fPrev,
                        double fNow, const std::vector<double> &gNow) {
  double ro;
  std::vector<double> xDif(xNow.size());

  if (_VecLen(gNow) >= H_EPS3)
    return false;

  _VecSub(xNow, xPrev, xDif);
  ro = _VecLen(xPrev);
  if (ro < H_EPS2)
    ro = 1.0;
  ro *= H_EPS1;
  if (_VecLen(xDif) >= ro)
    return false;

  ro = fabs(fPrev);
  if (ro < H_EPS2)
    ro = 1.0;
  ro *= H_EPS1;
  fNow -= fPrev;
  if (fabs(fNow) >= ro)
    return false;

  return true;
}

void AnalysisEqs(const std::vector<EqInfo> &eqTab, int eqNum,
                 std::vector<int> &eqHeads) {
  eqHeads.resize(eqNum);
  for (int i = 0; i < eqNum; i++) {
    const EqInfo &eq = eqTab[i];
    int left = eq._left;
    int right = eq._right;

    eqHeads[i] = left;
  }
}

double CalcSparsity(const array2<double> &H) {
  int n = H.rows();
  int m = H.cols();
  int zero = 0;
// parallel
#pragma omp parallel for reduction(+ : zero)
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (H(i, j) < epsZero1)
        zero++;
    }
  }
  return zero * 1.0 / (n * m);
}

double CalcSparsity(const std::vector<double> &y) {
  int n = y.size();
  int zero = 0;
#pragma omp parallel for reduction(+ : zero)
  for (int i = 0; i < n; i++) {
    if (y[i] < epsZero1)
      zero++;
  }
  return zero * 1.0 / n;
}

int BFGSSolveEqs(const std::string &filepath) {
  double eps = _GetEps() * _GetEps();
  int itMax = _GetMaxIt();

  double step = _GetStep();

  std::vector<double> xNow, xKeep;
  std::vector<int> varMap, revMap;
  std::vector<EqInfo> objEqs;
  int numObjEqs;
  std::vector<EqInfo> gradEqs;
  int numGradEqs;

  {
    FILE *fp = fopen(filepath.c_str(), "rb");
    if (fp == NULL) {
      std::cerr << filepath << " failed to open for read" << std::endl;
      // printf("data/bfgs.dat failed to open for read.\n");
      return false;
    }

    double t0 = omp_get_wtime();
    int nx;
    fread(&nx, sizeof(int), 1, fp);
    xNow.resize(nx);
    fread(xNow.data(), sizeof(double), nx, fp);

    int n1, no;
    fread(&n1, sizeof(int), 1, fp);
    fread(&no, sizeof(int), 1, fp);
    numObjEqs = no;
    objEqs.resize(n1);
    fread(objEqs.data(), sizeof(EqInfo), n1, fp);

    int ng;
    fread(&ng, sizeof(int), 1, fp);
    gradEqs.resize(ng);
    fread(gradEqs.data(), sizeof(EqInfo), ng, fp);
    numGradEqs = ng;

    int nk;
    fread(&nk, sizeof(int), 1, fp);
    assert(nk == nx);
    xKeep.resize(nk);
    fread(xKeep.data(), sizeof(double), nk, fp);

    double dt = omp_get_wtime() - t0;
    printf("###Data loading used %2.5f s ...\n", dt);

    // to remove recursive eval
    AnalysisEqs(objEqs, numObjEqs, objEqHeads);
    objEqVals.resize(objEqs.size());
    AnalysisEqs(gradEqs, nx, gradEqHeads);
    gradEqVals.resize(gradEqs.size());
  }

  double t0 = omp_get_wtime();
  // Do optimization
  double fNow = 0, fPrev = 0;
  int n = xNow.size();
  int k = 0; // useless?
  int itCounter = 0;

  std::vector<double> gPrev, gNow, xPrev, p, y, s, yTH, Hy;

  array2<double> H;

  xPrev = xNow;
  gPrev.resize(n);
  gNow.resize(n);
  p.resize(n);
  y.resize(n);
  s.resize(n);
  yTH.resize(n);
  Hy.resize(n);
  H.resize(n, n);

  // store the sparsity of H and the error of f
  std::vector<double> sparsity, fErr;

  // STEP1:
  fPrev = _CalcObj(xNow, objEqs, numObjEqs);
  _CalcGrad(xNow, gPrev, gradEqs);

STEP2:
  for (int i = 0; i < n; i++) {
    H(i, i) = 1.0;
    p[i] = -gPrev[i];
  }
  k = 0;
  _VecNorm(p);

STEP3:
  // START_CPU
  if (itCounter++ > itMax)
    goto END;

  xPrev = xNow;
  _LinearSearch(xPrev, p, step, xNow, objEqs, numObjEqs);
  fNow = _CalcObj(xNow, objEqs, numObjEqs);
  std::cout << itCounter << " iterations, "
            << "f(x) = " << fNow << std::endl;
  // fErr.push_back(fNow);
  // sparsity.push_back(CalcSparsity(H));

  if (fNow < eps)
    goto END;

  _CalcGrad(xNow, gNow, gradEqs);
  // END_CPU("STEP3")

  // STEP4:
  if (_HTerminate(xPrev, xNow, fPrev, fNow, gNow))
    goto END;

  // STEP5:
  if (fNow > fPrev) {
    _VecCopy(xNow, xPrev);
    goto STEP2;
  }

  // STEP6:
  if (k == n) {
    fPrev = fNow;
    _VecCopy(gPrev, gNow);
    goto STEP2;
  }

  // STEP7:
  // START_CPU
  _VecSub(gNow, gPrev, y);
  _VecSub(xNow, xPrev, s);

  // double sparsity_y = CalcSparsity(y);
  // printf("sparsity_y = %lf\n", sparsity_y);

  {
    double sy = _VecDot(s, y);
    if (fabs(sy) < epsZero1)
      goto END;

    _CalcyTH(y, H, yTH);
    _CalcHy(H, y, Hy);
    _UpdateH(H, yTH, y, sy, s, Hy);
    _Calcp(H, gNow, p);
    _VecNorm(p);

    fPrev = fNow;
    _VecCopy(gPrev, gNow);
    _VecCopy(xPrev, xNow);
    // END_CPU("STEP7")
    goto STEP3;
  }

END:
  std::cout << itCounter << " iterations" << std::endl;
  std::cout << "f(x) = " << fNow << std::endl;
  double dt = omp_get_wtime() - t0;
  printf("###Solver totally used %2.5f s ...\n", dt);
  // save the sparsity and fErr
  // FILE *fp = fopen("sparsity_1e-20.csv", "w");
  // fprintf(fp, "iter,sparsity,fErr\n");
  // for (int i = 0; i < sparsity.size(); i++) {
  //   fprintf(fp, "%d,%lf,%lf\n", i, sparsity[i], fErr[i]);
  // }
  // fclose(fp);

  // Put results back...
  if (fNow < eps) {
    printf("Solved!!!!\n");
    return true;
  } else {
    printf("Solver Failed!!!!\n");
    return false;
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <file_path>" << std::endl;
    return 1;
  }
  std::string filepath = argv[1];
  BFGSSolveEqs(filepath);
}
