#ifndef DEVICE_CODE_CDSORT_HPP
#define DEVICE_CODE_CDSORT_HPP

MYDEVFN unsigned dHZ_L0_s
(volatile double *const F,
 volatile double *const G,
 volatile double *const V,
 const unsigned x,
 const unsigned y)
{
  unsigned
    blk_transf_s = 0u,
    blk_transf_b = 0u,
    p = _strat0[0u][y][0u],
    q = _strat0[0u][y][1u];
  double
    App, Aqq, Bpp, Bqq,
    Fp_, Fq_, Gp_, Gq_, Vp_, Vq_;

  F32(V, x, p) = ((x == p) ? 1.0 : 0.0);
  F32(V, x, q) = ((x == q) ? 1.0 : 0.0);

  __syncthreads();

  for (unsigned swp = 0u; swp < _nSwp; ++swp) {
    int
      swp_transf_s = 0,
      swp_transf_b = 0;

    for (unsigned step = 0u; step < _STRAT0_STEPS; ++step) {
      p = _strat0[step][y][0u];
      q = _strat0[step][y][1u];

      const double Fp = F32(F, x, p);
      const double Fq = F32(F, x, q);

      const double Gp = F32(G, x, p);
      const double Gq = F32(G, x, q);

      const double Vp = F32(V, x, p);
      const double Vq = F32(V, x, q);

      __syncthreads();

      App = dSsq32(Fp);
      Aqq = dSsq32(Fq);
      Bpp = dSsq32(Gp);
      Bqq = dSsq32(Gq);
      double Apq = dSum32(Fp * Fq);
      double Bpq = dSum32(Gp * Gq);

      if (Bpp != 1.0) {
        App = __ddiv_rn(App, Bpp);
        Bpp = my_drsqrt_rn(Bpp);
        Apq *= Bpp;
        Bpq *= Bpp;
      }

      if (Bqq != 1.0) {
        Aqq = __ddiv_rn(Aqq, Bqq);
        Bqq = my_drsqrt_rn(Bqq);
        Apq *= Bqq;
        Bpq *= Bqq;
      }

      const double Bpq_ = fabs(Bpq);
      const int transf_s = (!(Bpq_ < HZ_MYTOL) ? 1 :
                            !(fabs(Apq) < ((__dsqrt_rn(App) * __dsqrt_rn(Aqq)) * HZ_MYTOL)));
      int transf_b = 0;

      Fp_ = Fp; Fq_ = Fq;
      Gp_ = Gp; Gq_ = Gq;
      Vp_ = Vp; Vq_ = Vq;

      swp_transf_s += (__syncthreads_count(transf_s) >> WARP_SZ_LGi);
      if (transf_s) {
        double CosF, SinF, CosP, SinP;

        dRot(App, Aqq, Apq, Bpq, Bpq_, CosF, SinF, CosP, SinP);
        transf_b = ((CosF != 1.0) || (CosP != 1.0));

        if (Bpp != 1.0) {
          CosF *= Bpp;
          SinF *= Bpp;
        }
        if (Bqq != 1.0) {
          CosP *= Bqq;
          SinP *= Bqq;
        }
        const int
          fn1 = (CosF != 1.0),
          pn1 = (CosP != 1.0);

        if (fn1 || pn1) {
          if (App >= Aqq) {
            if (fn1) {
              if (SinP == 1.0) {
                Fp_ = __fma_rn(CosF, Fp, -Fq);
                Gp_ = __fma_rn(CosF, Gp, -Gq);
                Vp_ = __fma_rn(CosF, Vp, -Vq);
              }
              else if (SinP == -1.0) {
                Fp_ = __fma_rn(CosF, Fp, Fq);
                Gp_ = __fma_rn(CosF, Gp, Gq);
                Vp_ = __fma_rn(CosF, Vp, Vq);
              }
              else {
                Fp_ = CosF * Fp - SinP * Fq;
                Gp_ = CosF * Gp - SinP * Gq;
                Vp_ = CosF * Vp - SinP * Vq;
              }
            }
            else {
              const double SinP_ = -SinP;
              Fp_ = __fma_rn(SinP_, Fq, Fp);
              Gp_ = __fma_rn(SinP_, Gq, Gp);
              Vp_ = __fma_rn(SinP_, Vq, Vp);
            }
            if (pn1) {
              if (SinF == 1.0) {
                Fq_ = __fma_rn(CosP, Fq, Fp);
                Gq_ = __fma_rn(CosP, Gq, Gp);
                Vq_ = __fma_rn(CosP, Vq, Vp);
              }
              else if (SinF == -1.0) {
                Fq_ = __fma_rn(CosP, Fq, -Fp);
                Gq_ = __fma_rn(CosP, Gq, -Gp);
                Vq_ = __fma_rn(CosP, Vq, -Vp);
              }
              else {
                Fq_ = SinF * Fp + CosP * Fq;
                Gq_ = SinF * Gp + CosP * Gq;
                Vq_ = SinF * Vp + CosP * Vq;
              }
            }
            else {
              Fq_ = __fma_rn(SinF, Fp, Fq);
              Gq_ = __fma_rn(SinF, Gp, Gq);
              Vq_ = __fma_rn(SinF, Vp, Vq);
            }
          }
          else {
            if (fn1) {
              if (SinP == 1.0) {
                Fq_ = __fma_rn(CosF, Fp, -Fq);
                Gq_ = __fma_rn(CosF, Gp, -Gq);
                Vq_ = __fma_rn(CosF, Vp, -Vq);
              }
              else if (SinP == -1.0) {
                Fq_ = __fma_rn(CosF, Fp, Fq);
                Gq_ = __fma_rn(CosF, Gp, Gq);
                Vq_ = __fma_rn(CosF, Vp, Vq);
              }
              else {
                Fq_ = CosF * Fp - SinP * Fq;
                Gq_ = CosF * Gp - SinP * Gq;
                Vq_ = CosF * Vp - SinP * Vq;
              }
            }
            else {
              const double SinP_ = -SinP;
              Fq_ = __fma_rn(SinP_, Fq, Fp);
              Gq_ = __fma_rn(SinP_, Gq, Gp);
              Vq_ = __fma_rn(SinP_, Vq, Vp);
            }
            if (pn1) {
              if (SinF == 1.0) {
                Fp_ = __fma_rn(CosP, Fq, Fp);
                Gp_ = __fma_rn(CosP, Gq, Gp);
                Vp_ = __fma_rn(CosP, Vq, Vp);
              }
              else if (SinF == -1.0) {
                Fp_ = __fma_rn(CosP, Fq, -Fp);
                Gp_ = __fma_rn(CosP, Gq, -Gp);
                Vp_ = __fma_rn(CosP, Vq, -Vp);
              }
              else {
                Fp_ = SinF * Fp + CosP * Fq;
                Gp_ = SinF * Gp + CosP * Gq;
                Vp_ = SinF * Vp + CosP * Vq;
              }
            }
            else {
              Fp_ = __fma_rn(SinF, Fp, Fq);
              Gp_ = __fma_rn(SinF, Gp, Gq);
              Vp_ = __fma_rn(SinF, Vp, Vq);
            }
          }
        }
        else {
          const double SinP_ = -SinP;
          if (App >= Aqq) {
            Fp_ = __fma_rn(SinP_, Fq, Fp);
            Fq_ = __fma_rn(SinF, Fp, Fq);
            Gp_ = __fma_rn(SinP_, Gq, Gp);
            Gq_ = __fma_rn(SinF, Gp,  Gq);
            Vp_ = __fma_rn(SinP_, Vq, Vp);
            Vq_ = __fma_rn(SinF, Vp, Vq);
          }
          else {
            Fq_ = __fma_rn(SinP_, Fq, Fp);
            Fp_ = __fma_rn(SinF, Fp, Fq);
            Gq_ = __fma_rn(SinP_, Gq, Gp);
            Gp_ = __fma_rn(SinF, Gp,  Gq);
            Vq_ = __fma_rn(SinP_, Vq, Vp);
            Vp_ = __fma_rn(SinF, Vp, Vq);
          }
        }
        F32(F, x, p) = Fp_;
        F32(F, x, q) = Fq_;
        F32(G, x, p) = Gp_;
        F32(G, x, q) = Gq_;
        F32(V, x, p) = Vp_;
        F32(V, x, q) = Vq_;
      }
      else if (App < Aqq) {
        Fp_ = Fq;
        Fq_ = Fp;
        Gp_ = Gq;
        Gq_ = Gp;
        Vp_ = Vq;
        Vq_ = Vp;
        F32(F, x, p) = Fp_;
        F32(F, x, q) = Fq_;
        F32(G, x, p) = Gp_;
        F32(G, x, q) = Gq_;
        F32(V, x, p) = Vp_;
        F32(V, x, q) = Vq_;
      }

      swp_transf_b += (__syncthreads_count(transf_b) >> WARP_SZ_LGi);
    }

    if (swp_transf_s) {
      blk_transf_s += static_cast<unsigned>(swp_transf_s);
      blk_transf_b += static_cast<unsigned>(swp_transf_b);
    }
    else
      break;
  }

  if (blk_transf_s) {
    // normalize V

    App = dSsq32(Fp_);
    Bpp = dSsq32(Gp_);
    const double Vpp_ = my_drsqrt_rn(App + Bpp);
    if (Vpp_ != 1.0)
      F32(V, x, p) = Vp_ * Vpp_;

    Aqq = dSsq32(Fq_);
    Bqq = dSsq32(Gq_);
    const double Vqq_ = my_drsqrt_rn(Aqq + Bqq);
    if (Vqq_ != 1.0)
      F32(V, x, q) = Vq_ * Vqq_;

    if (!y && !x) {
      const unsigned bix2 = (unsigned)(blockIdx.x) << 1u;
      ((unsigned long long*)_S)[bix2] += blk_transf_s;
      if (blk_transf_b)
        ((unsigned long long*)_S)[bix2 + 1u] += blk_transf_b;
    }
  }

  __syncthreads();
  return blk_transf_s;
}

#endif // !DEVICE_CODE_CDSORT_HPP
