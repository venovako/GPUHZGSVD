#ifndef DEVICE_CODE_COMMON_QR_HPP
#define DEVICE_CODE_COMMON_QR_HPP

// [  C S ] [ F ] = [ R ]
// [ -S C ] [ G ]   [ 0 ]
// R >= 0
MYDEVFN void dGivens
(const double f,
 const double g,
 double &c,
 double &s,
 double &r)
{
  const double f_ = fabs(f);
  const double g_ = fabs(g);
  if (f_ >= g_) {
    if (g_ == 0.0) {
      c = copysign(1.0, f);
      s = 0.0;
      r = f_;
    }
    else {
      const double g_f = _ddiv_rn(g_, f_);
      r = _dmul_rn(f_, _dsqrt_rn(_fma_rn(g_f, g_f, 1.0)));
      c = _ddiv_rn(f, r);
      s = _ddiv_rn(g, r);
    }
  }
  else {
    if (f_ == 0.0) {
      c = 0.0;
      s = copysign(1.0, g);
      r = g_;
    }
    else {
      const double f_g = _ddiv_rn(f_, g_);
      r = _dmul_rn(g_, _dsqrt_rn(_fma_rn(f_g, f_g, 1.0)));
      c = _ddiv_rn(f, r);
      s = _ddiv_rn(g, r);
    }
  }
}

MYDEVFN void dQR32
(volatile double *const __restrict__ A,
 const unsigned x,
 const unsigned y0,
 const unsigned y1)
{
  #pragma unroll
  for (unsigned k = 0u; k < 31u; ++k) {
    double nrm2 = 0.0, Axk, Axk_, beta, alpha_beta, tau;
    // compute the Householer reflector, see DLARFG
    const unsigned my0 = k + y0;
    if (my0 < 32u) {
      Axk = F32(A, x, k);
      Axk_ = ((x >= k) ? Axk : 0.0);
      // TODO: use dNRM2_32 from GPUJACHx instead
      nrm2 = _dsqrt_rn(dSsq32(Axk_));
      if (nrm2 > 0.0) {
        const double alpha = F32(A, k, k);
        beta = copysign(nrm2, alpha);
        alpha_beta = _dadd_rn(alpha, beta);
        tau = _ddiv_rn(alpha_beta, beta);
      }
    }
    __syncthreads();
    if (nrm2 > 0.0) {
      // apply the Householder reflector
      if (my0 == k)
        F32(A, x, k) = ((x == k) ? -beta : ((x > k) ? 0.0 : Axk));
      else {
        const double Axy = F32(A, x, my0);
        const double Vxk = ((x == k) ? 1.0 : ((x > k) ? _ddiv_rn(Axk_, alpha_beta) : 0.0));
        const double dp = dSum32(_dmul_rn(Vxk, Axy));
        const double _tdp = -_dmul_rn(tau, dp);
        F32(A, x, my0) = _fma_rn(_tdp, Vxk, Axy);
      }
      const unsigned my1 = k + y1;
      if (my1 < 32u) {
        const double Axy = F32(A, x, my1);
        const double Vxk = ((x == k) ? 1.0 : ((x > k) ? _ddiv_rn(Axk_, alpha_beta) : 0.0));
        const double dp = dSum32(_dmul_rn(Vxk, Axy));
        const double _tdp = -_dmul_rn(tau, dp);
        F32(A, x, my1) = _fma_rn(_tdp, Vxk, Axy);
      }
    }
    __syncthreads();
  }
}

MYDEVFN void dPeelOff
(volatile double *const __restrict__ R0,
 volatile double *const __restrict__ R1,
 const unsigned x,
 const unsigned y0,
 const unsigned y1)
{
  #pragma unroll
  for (unsigned k = 0u; k < 32u; ++k) {
    const unsigned
      my0 = (k + y0),
      my1 = (k + y1);
    unsigned x_k;
    double c, s, r, a_, b_;
    bool store = false;
    const bool active = ((x >= k) && (x <= my1));
    if ((my0 < 32u) && active) {
      x_k = x - k;
      const double f = F32(R0, x, x);
      const double g = F32(R1, x_k, x);
      dGivens(f, g, c, s, r);
      if (x < my0) {
        const double a = F32(R0, x, my0);
        const double b = F32(R1, x_k, my0);
        a_ = _fma_rn(c, a,  _dmul_rn(s, b));
        b_ = _fma_rn(c, b, -_dmul_rn(s, a));
        store = true;
      }
      else if (x == my0) {
        a_ = r;
        b_ = 0.0;
        store = true;
      }
    }
    __syncthreads();
    if (store) {
      F32(R0, x, my0) = a_;
      F32(R1, x_k, my0) = b_;
      store = false;
    }
    __syncthreads();
    if ((my1 < 32u) && active) {
      if (x < my1) {
        const double a = F32(R0, x, my1);
        const double b = F32(R1, x_k, my1);
        a_ = _fma_rn(c, a,  _dmul_rn(s, b));
        b_ = _fma_rn(c, b, -_dmul_rn(s, a));
        store = true;
      }
      else if (x == my1) {
        a_ = r;
        b_ = 0.0;
        store = true;
      }
    }
    __syncthreads();
    if (store) {
      F32(R0, x, my1) = a_;
      F32(R1, x_k, my1) = b_;
    }
    __syncthreads();
  }
}

MYDEVFN void dFactorize
(const double *const __restrict__ A0,
 const double *const __restrict__ A1,
 volatile double *const __restrict__ A,
 volatile double *const __restrict__ V,
 const unsigned m,
 const unsigned x,
 const unsigned y0,
 const unsigned y1)
{
  F32(A, x, y0) = A0[x];
  F32(A, x, y1) = A1[x];
  __syncthreads();
  dQR32(A, x, y0, y1);

  for (unsigned i = x + 32u; i < m; i += 32u) {
    F32(V, x, y0) = A0[i];
    F32(V, x, y1) = A1[i];
    __syncthreads();
    dQR32(V, x, y0, y1);
    dPeelOff(A, V, x, y0, y1);
  }
}

#endif /* !DEVICE_CODE_COMMON_QR_HPP */
