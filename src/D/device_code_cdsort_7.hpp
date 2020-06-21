#ifndef DEVICE_CODE_CDSORT_HPP
#define DEVICE_CODE_CDSORT_HPP

MYDEVFN unsigned dHZ_L0_sv
(volatile double *const __restrict__ F,
 volatile double *const __restrict__ G,
 volatile double *const __restrict__ V,
 const unsigned x,
 const unsigned y)
{
  unsigned
    blk_transf_s = 0u,
    blk_transf_b = 0u,
    p = _strat0[0u][y][0u],
    q = _strat0[0u][y][1u];
  double
    App, Aqq, Bpp, Bqq, s0, s1,
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

      App = dSSQ32(Fp, s0, s1);
      Aqq = dSSQ32(Fq, s0, s1);
      Bpp = dSSQ32(Gp, s0, s1);
      Bqq = dSSQ32(Gq, s0, s1);
      double Apq = dSUM_PROD_32(Fp, Fq, s0, s1);
      double Bpq = dSUM_PROD_32(Gp, Gq, s0, s1);

      if (Bpp != 1.0) {
        App = _ddiv_rn(App, Bpp);
        Bpp = _drsqrt_rn(Bpp);
        Apq = _dmul_rn(Apq, Bpp);
        Bpq = _dmul_rn(Bpq, Bpp);
      }

      if (Bqq != 1.0) {
        Aqq = _ddiv_rn(Aqq, Bqq);
        Bqq = _drsqrt_rn(Bqq);
        Apq = _dmul_rn(Apq, Bqq);
        Bpq = _dmul_rn(Bpq, Bqq);
      }

      const double Bpq_ = fabs(Bpq);
      const int transf_s = (!(Bpq_ < HZ_MYTOL) ? 1 :
                            !(fabs(Apq) < _dmul_rn(_dmul_rn(_dsqrt_rn(App), _dsqrt_rn(Aqq)), HZ_MYTOL)));
      int transf_b = 0;

      Fp_ = Fp; Fq_ = Fq;
      Gp_ = Gp; Gq_ = Gq;
      Vp_ = Vp; Vq_ = Vq;

      swp_transf_s += (__syncthreads_count(transf_s) >> WARP_SZ_LGi);
      if (transf_s) {
        double CosF, SinF, CosP, SinP;
        int fn1, pn1;

        transf_b = dRot(App, Aqq, Apq, Bpq, Bpq_, CosF, SinF, CosP, SinP, fn1, pn1);

        if (Bpp != 1.0) {
          CosF = _dmul_rn(CosF, Bpp);
          SinF = _dmul_rn(SinF, Bpp);
        }
        if (Bqq != 1.0) {
          CosP = _dmul_rn(CosP, Bqq);
          SinP = _dmul_rn(SinP, Bqq);
        }
        fn1 = (CosF != 1.0);
        pn1 = (CosP != 1.0);

        if (fn1 || pn1) {
          if (App >= Aqq) {
            if (fn1) {
              if (SinP == 1.0) {
                Fp_ = _fma_rn(CosF, Fp, -Fq);
                Gp_ = _fma_rn(CosF, Gp, -Gq);
                Vp_ = _fma_rn(CosF, Vp, -Vq);
              }
              else if (SinP == -1.0) {
                Fp_ = _fma_rn(CosF, Fp, Fq);
                Gp_ = _fma_rn(CosF, Gp, Gq);
                Vp_ = _fma_rn(CosF, Vp, Vq);
              }
              else {
                Fp_ = _fma_rn(CosF, Fp, -_dmul_rn(SinP, Fq));
                Gp_ = _fma_rn(CosF, Gp, -_dmul_rn(SinP, Gq));
                Vp_ = _fma_rn(CosF, Vp, -_dmul_rn(SinP, Vq));
              }
            }
            else {
              const double SinP_ = -SinP;
              Fp_ = _fma_rn(SinP_, Fq, Fp);
              Gp_ = _fma_rn(SinP_, Gq, Gp);
              Vp_ = _fma_rn(SinP_, Vq, Vp);
            }
            if (pn1) {
              if (SinF == 1.0) {
                Fq_ = _fma_rn(CosP, Fq, Fp);
                Gq_ = _fma_rn(CosP, Gq, Gp);
                Vq_ = _fma_rn(CosP, Vq, Vp);
              }
              else if (SinF == -1.0) {
                Fq_ = _fma_rn(CosP, Fq, -Fp);
                Gq_ = _fma_rn(CosP, Gq, -Gp);
                Vq_ = _fma_rn(CosP, Vq, -Vp);
              }
              else {
                Fq_ = _fma_rn(SinF, Fp, _dmul_rn(CosP, Fq));
                Gq_ = _fma_rn(SinF, Gp, _dmul_rn(CosP, Gq));
                Vq_ = _fma_rn(SinF, Vp, _dmul_rn(CosP, Vq));
              }
            }
            else {
              Fq_ = _fma_rn(SinF, Fp, Fq);
              Gq_ = _fma_rn(SinF, Gp, Gq);
              Vq_ = _fma_rn(SinF, Vp, Vq);
            }
          }
          else {
            if (fn1) {
              if (SinP == 1.0) {
                Fq_ = _fma_rn(CosF, Fp, -Fq);
                Gq_ = _fma_rn(CosF, Gp, -Gq);
                Vq_ = _fma_rn(CosF, Vp, -Vq);
              }
              else if (SinP == -1.0) {
                Fq_ = _fma_rn(CosF, Fp, Fq);
                Gq_ = _fma_rn(CosF, Gp, Gq);
                Vq_ = _fma_rn(CosF, Vp, Vq);
              }
              else {
                Fq_ = _fma_rn(CosF, Fp, -_dmul_rn(SinP, Fq));
                Gq_ = _fma_rn(CosF, Gp, -_dmul_rn(SinP, Gq));
                Vq_ = _fma_rn(CosF, Vp, -_dmul_rn(SinP, Vq));
              }
            }
            else {
              const double SinP_ = -SinP;
              Fq_ = _fma_rn(SinP_, Fq, Fp);
              Gq_ = _fma_rn(SinP_, Gq, Gp);
              Vq_ = _fma_rn(SinP_, Vq, Vp);
            }
            if (pn1) {
              if (SinF == 1.0) {
                Fp_ = _fma_rn(CosP, Fq, Fp);
                Gp_ = _fma_rn(CosP, Gq, Gp);
                Vp_ = _fma_rn(CosP, Vq, Vp);
              }
              else if (SinF == -1.0) {
                Fp_ = _fma_rn(CosP, Fq, -Fp);
                Gp_ = _fma_rn(CosP, Gq, -Gp);
                Vp_ = _fma_rn(CosP, Vq, -Vp);
              }
              else {
                Fp_ = _fma_rn(SinF, Fp, _dmul_rn(CosP, Fq));
                Gp_ = _fma_rn(SinF, Gp, _dmul_rn(CosP, Gq));
                Vp_ = _fma_rn(SinF, Vp, _dmul_rn(CosP, Vq));
              }
            }
            else {
              Fp_ = _fma_rn(SinF, Fp, Fq);
              Gp_ = _fma_rn(SinF, Gp, Gq);
              Vp_ = _fma_rn(SinF, Vp, Vq);
            }
          }
        }
        else {
          const double SinP_ = -SinP;
          if (App >= Aqq) {
            Fp_ = _fma_rn(SinP_, Fq, Fp);
            Fq_ = _fma_rn(SinF, Fp, Fq);
            Gp_ = _fma_rn(SinP_, Gq, Gp);
            Gq_ = _fma_rn(SinF, Gp,  Gq);
            Vp_ = _fma_rn(SinP_, Vq, Vp);
            Vq_ = _fma_rn(SinF, Vp, Vq);
          }
          else {
            Fq_ = _fma_rn(SinP_, Fq, Fp);
            Fp_ = _fma_rn(SinF, Fp, Fq);
            Gq_ = _fma_rn(SinP_, Gq, Gp);
            Gp_ = _fma_rn(SinF, Gp,  Gq);
            Vq_ = _fma_rn(SinP_, Vq, Vp);
            Vp_ = _fma_rn(SinF, Vp, Vq);
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

  // normalize V

  App = dSSQ32(Fp_, s0, s1);
  Bpp = dSSQ32(Gp_, s0, s1);
  const double Vpp_ = _drsqrt_rn(_dadd_rn(App, Bpp));
  if (Vpp_ != 1.0)
    F32(V, x, p) = _dmul_rn(Vp_, Vpp_);

  Aqq = dSSQ32(Fq_, s0, s1);
  Bqq = dSSQ32(Gq_, s0, s1);
  const double Vqq_ = _drsqrt_rn(_dadd_rn(Aqq, Bqq));
  if (Vqq_ != 1.0)
    F32(V, x, q) = _dmul_rn(Vq_, Vqq_);

  if (!y && !x) {
    const unsigned bix2 = (unsigned)(blockIdx.x) << C_SHIFTR;
    _C[bix2 + C_SMALL] += blk_transf_s;
    if (blk_transf_b)
      _C[bix2 + C_BIG] += blk_transf_b;
  }

  __syncthreads();
  return blk_transf_s;
}

#endif /* !DEVICE_CODE_CDSORT_HPP */
