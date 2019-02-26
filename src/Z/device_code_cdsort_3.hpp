#ifndef DEVICE_CODE_CDSORT_HPP
#define DEVICE_CODE_CDSORT_HPP

MYDEVFN unsigned zHZ_L0_sv
(volatile cuD *const FD, volatile cuJ *const FJ,
 volatile cuD *const GD, volatile cuJ *const GJ,
 volatile cuD *const VD, volatile cuJ *const VJ,
 const unsigned x,
 const unsigned y)
{
  unsigned
    blk_transf_s = 0u,
    blk_transf_b = 0u,
    p = _strat0[0u][y][0u],
    q = _strat0[0u][y][1u];
  cuD
    App, Aqq, Bpp, Bqq, Vpp, Vqq;
  cuD
    FpD, FqD, GpD, GqD, VpD, VqD,
    Fp_D, Fq_D, Gp_D, Gq_D, Vp_D, Vq_D,
    ApqD, BpqD;
  cuJ
    FpJ, FqJ, GpJ, GqJ, VpJ, VqJ,
    Fp_J, Fq_J, Gp_J, Gq_J, Vp_J, Vq_J,
    ApqJ, BpqJ;

  F32(VD, x, p) = ((x == p) ? 1.0 : 0.0);
  F32(VJ, x, p) = 0.0;
  F32(VD, x, q) = ((x == q) ? 1.0 : 0.0);
  F32(VJ, x, q) = 0.0;
  __syncthreads();

  for (unsigned swp = 0u; swp < _nSwp; ++swp) {
    int
      swp_transf_s = 0,
      swp_transf_b = 0;

    for (unsigned step = 0u; step < _STRAT0_STEPS; ++step) {
      p = _strat0[step][y][0u];
      q = _strat0[step][y][1u];

      FpD = F32(FD, x, p);
      FpJ = F32(FJ, x, p);
      FqD = F32(FD, x, q);
      FqJ = F32(FJ, x, q);

      GpD = F32(GD, x, p);
      GpJ = F32(GJ, x, p);
      GqD = F32(GD, x, q);
      GqJ = F32(GJ, x, q);

      VpD = F32(VD, x, p);
      VpJ = F32(VJ, x, p);
      VqD = F32(VD, x, q);
      VqJ = F32(VJ, x, q);

      __syncthreads();

      App = zSsq32(FpD, FpJ);
      assert(App > 0.0);
      assert(App < INFTY);

      Aqq = zSsq32(FqD, FqJ);
      assert(Aqq > 0.0);
      assert(Aqq < INFTY);

      Bpp = zSsq32(GpD, GpJ);
      assert(Bpp > 0.0);
      assert(Bpp < INFTY);

      Bqq = zSsq32(GqD, GqJ);
      assert(Bqq > 0.0);
      assert(Bqq < INFTY);

      zDot32(ApqD, ApqJ, FpD, FpJ, FqD, FqJ);
      zDot32(BpqD, BpqJ, GpD, GpJ, GqD, GqJ);

      if (Bpp != 1.0) {
        App = __ddiv_rn(App, Bpp);
        Bpp = my_drsqrt_rn(Bpp);
        ApqD *= Bpp;
        ApqJ *= Bpp;
        BpqD *= Bpp;
        BpqJ *= Bpp;
      }

      if (Bqq != 1.0) {
        Aqq = __ddiv_rn(Aqq, Bqq);
        Bqq = my_drsqrt_rn(Bqq);
        ApqD *= Bqq;
        ApqJ *= Bqq;
        BpqD *= Bqq;
        BpqJ *= Bqq;
      }

      const double Bpq_ = hypot(BpqD, BpqJ);
      assert(Bpq_ < 1.0);

      const int transf_s = (!(Bpq_ < HZ_MYTOL) ? 1 :
                            !(hypot(ApqD, ApqJ) < ((__dsqrt_rn(App) * __dsqrt_rn(Aqq)) * HZ_MYTOL)));
      int transf_b = 0;

      swp_transf_s += (__syncthreads_count(transf_s) >> WARP_SZ_LGi);
      if (transf_s) {
        double CosF, CosP;
        cuD SinFD, _SinPD;
        cuJ SinFJ, _SinPJ;

        zRot(App, Aqq, ApqD, ApqJ, BpqD, BpqJ, Bpq_, CosF, SinFD, SinFJ, CosP, _SinPD, _SinPJ);
        const int
          fn1 = (CosF != 1.0),
          pn1 = (CosP != 1.0);
        transf_b = (fn1 || pn1);

        if (fn1) {
          if (pn1) {
            Zfma(Fp_D, Fp_J, FqD, FqJ, _SinPD, _SinPJ, FpD * CosF, FpJ * CosF);
            Zfma(Fq_D, Fq_J, FpD, FpJ, SinFD, SinFJ, FqD * CosP, FqJ * CosP);

            Zfma(Gp_D, Gp_J, GqD, GqJ, _SinPD, _SinPJ, GpD * CosF, GpJ * CosF);
            Zfma(Gq_D, Gq_J, GpD, GpJ, SinFD, SinFJ, GqD * CosP, GqJ * CosP);

            Zfma(Vp_D, Vp_J, VqD, VqJ, _SinPD, _SinPJ, VpD * CosF, VpJ * CosF);
            Zfma(Vq_D, Vq_J, VpD, VpJ, SinFD, SinFJ, VqD * CosP, VqJ * CosP);
          }
          else {
            Zfma(Fp_D, Fp_J, FqD, FqJ, _SinPD, _SinPJ, FpD * CosF, FpJ * CosF);
            Zfma(Fq_D, Fq_J, FpD, FpJ, SinFD, SinFJ, FqD, FqJ);

            Zfma(Gp_D, Gp_J, GqD, GqJ, _SinPD, _SinPJ, GpD * CosF, GpJ * CosF);
            Zfma(Gq_D, Gq_J, GpD, GpJ, SinFD, SinFJ, GqD, GqJ);

            Zfma(Vp_D, Vp_J, VqD, VqJ, _SinPD, _SinPJ, VpD * CosF, VpJ * CosF);
            Zfma(Vq_D, Vq_J, VpD, VpJ, SinFD, SinFJ, VqD, VqJ);
          }
        }
        else {
          if (pn1) {
            Zfma(Fp_D, Fp_J, FqD, FqJ, _SinPD, _SinPJ, FpD, FpJ);
            Zfma(Fq_D, Fq_J, FpD, FpJ, SinFD, SinFJ, FqD * CosP, FqJ * CosP);

            Zfma(Gp_D, Gp_J, GqD, GqJ, _SinPD, _SinPJ, GpD, GpJ);
            Zfma(Gq_D, Gq_J, GpD, GpJ, SinFD, SinFJ, GqD * CosP, GqJ * CosP);

            Zfma(Vp_D, Vp_J, VqD, VqJ, _SinPD, _SinPJ, VpD, VpJ);
            Zfma(Vq_D, Vq_J, VpD, VpJ, SinFD, SinFJ, VqD * CosP, VqJ * CosP);
          }
          else {
            Zfma(Fp_D, Fp_J, FqD, FqJ, _SinPD, _SinPJ, FpD, FpJ);
            Zfma(Fq_D, Fq_J, FpD, FpJ, SinFD, SinFJ, FqD, FqJ);

            Zfma(Gp_D, Gp_J, GqD, GqJ, _SinPD, _SinPJ, GpD, GpJ);
            Zfma(Gq_D, Gq_J, GpD, GpJ, SinFD, SinFJ, GqD, GqJ);

            Zfma(Vp_D, Vp_J, VqD, VqJ, _SinPD, _SinPJ, VpD, VpJ);
            Zfma(Vq_D, Vq_J, VpD, VpJ, SinFD, SinFJ, VqD, VqJ);
          }
        }

        App = zSsq32(Fp_D, Fp_J);
        assert(App > 0.0);
        assert(App < INFTY);

        Aqq = zSsq32(Fq_D, Fq_J);
        assert(Aqq > 0.0);
        assert(Aqq < INFTY);

        if (App >= Aqq) {
          F32(FD, x, p) = Fp_D;
          F32(FJ, x, p) = Fp_J;

          F32(GD, x, p) = Gp_D;
          F32(GJ, x, p) = Gp_J;

          F32(VD, x, p) = Vp_D;
          F32(VJ, x, p) = Vp_J;

          F32(FD, x, q) = Fq_D;
          F32(FJ, x, q) = Fq_J;

          F32(GD, x, q) = Gq_D;
          F32(GJ, x, q) = Gq_J;

          F32(VD, x, q) = Vq_D;
          F32(VJ, x, q) = Vq_J;
        }
        else { // swap
          FpD = Fp_D; Fp_D = Fq_D; Fq_D = FpD;
          FpJ = Fp_J; Fp_J = Fq_J; Fq_J = FpJ;

          GpD = Gp_D; Gp_D = Gq_D; Gq_D = GpD;
          GpJ = Gp_J; Gp_J = Gq_J; Gq_J = GpJ;

          VpD = Vp_D; Vp_D = Vq_D; Vq_D = VpD;
          VpJ = Vp_J; Vp_J = Vq_J; Vq_J = VpJ;

          F32(FD, x, p) = Fp_D;
          F32(FJ, x, p) = Fp_J;

          F32(GD, x, p) = Gp_D;
          F32(GJ, x, p) = Gp_J;

          F32(VD, x, p) = Vp_D;
          F32(VJ, x, p) = Vp_J;

          F32(FD, x, q) = Fq_D;
          F32(FJ, x, q) = Fq_J;

          F32(GD, x, q) = Gq_D;
          F32(GJ, x, q) = Gq_J;

          F32(VD, x, q) = Vq_D;
          F32(VJ, x, q) = Vq_J;
        }
      }
      else if (App < Aqq) { // swap
        Fp_D = FqD; Fp_J = FqJ;
        Fq_D = FpD; Fq_J = FpJ;
        Gp_D = GqD; Gp_J = GqJ;
        Gq_D = GpD; Gq_J = GpJ;
        Vp_D = VqD; Vp_J = VqJ;
        Vq_D = VpD; Vq_J = VpJ;

        F32(FD, x, p) = Fp_D;
        F32(FJ, x, p) = Fp_J;

        F32(GD, x, p) = Gp_D;
        F32(GJ, x, p) = Gp_J;

        F32(VD, x, p) = Vp_D;
        F32(VJ, x, p) = Vp_J;

        F32(FD, x, q) = Fq_D;
        F32(FJ, x, q) = Fq_J;

        F32(GD, x, q) = Gq_D;
        F32(GJ, x, q) = Gq_J;

        F32(VD, x, q) = Vq_D;
        F32(VJ, x, q) = Vq_J;
      }
      else { // no transf.
        Fp_D = FpD; Fp_J = FpJ;
        Fq_D = FqD; Fq_J = FqJ;
        Gp_D = GpD; Gp_J = GpJ;
        Gq_D = GqD; Gq_J = GqJ;
        Vp_D = VpD; Vp_J = VpJ;
        Vq_D = VqD; Vq_J = VqJ;
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

  App = zSsq32(Fp_D, Fp_J);
  assert(App > 0.0);
  assert(App < INFTY);

  Bpp = zSsq32(Gp_D, Gp_J);
  assert(Bpp > 0.0);
  assert(Bpp < INFTY);

  Vpp = my_drsqrt_rn(App + Bpp);
  assert(Vpp > 0.0);
  assert(Vpp < INFTY);

  if (Vpp != 1.0) {
    F32(VD, x, p) *= Vpp;
    F32(VJ, x, p) *= Vpp;
  }
  __syncthreads();

  Aqq = zSsq32(Fq_D, Fq_J);
  assert(Aqq > 0.0);
  assert(Aqq < INFTY);

  Bqq = zSsq32(Gq_D, Gq_J);
  assert(Bqq > 0.0);
  assert(Bqq < INFTY);

  Vqq = my_drsqrt_rn(Aqq + Bqq);
  assert(Vqq > 0.0);
  assert(Vqq < INFTY);

  if (Vqq != 1.0) {
    F32(VD, x, q) *= Vqq;
    F32(VJ, x, q) *= Vqq;
  }
  __syncthreads();

  if (!y && !x) {
    const unsigned bix2 = (unsigned)(blockIdx.x) << 1u;
    ((unsigned long long*)_S)[bix2] += blk_transf_s;
    if (blk_transf_b)
      ((unsigned long long*)_S)[bix2 + 1u] += blk_transf_b;
  }

  __syncthreads();
  return blk_transf_s;
}

#endif // !DEVICE_CODE_CDSORT_HPP
