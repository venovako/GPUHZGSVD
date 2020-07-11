#ifndef DEVICE_CODE_COMMON_ROTATE_HPP
#define DEVICE_CODE_COMMON_ROTATE_HPP

MYDEVFN void dRot
(double &App,
 double &Aqq,
 const double Apq,
 const double Bpq,
 const double Bpq_,
 double &CosF,
 double &SinF,
 double &CosP,
 double &SinP)
{
  const double
    E = _dsub_rn(Aqq, App),
    V = _fma_rn(Bpq, -_dadd_rn(App, Aqq), _dmul_rn(Apq, 2.0));

  if ((V == 0.0) && (E == 0.0)) {
    const double S1 = _drsqrt_rn(_dadd_rn(1.0, Bpq_));
    const double S2 = _drsqrt_rn(_dsub_rn(1.0, Bpq_));
    const double CG = RSQRT_2;
    const double SG = -CG;

    CosF = _dmul_rn(CG, S1);
    SinP = _dmul_rn(SG, S1);
    SinF = _dmul_rn(SG, S2);
    CosP = _dmul_rn(CG, S2);
  }
  else {
    const double
      Bpqp = _dsqrt_rn(_dadd_rn(1.0, Bpq)),
      Bpqm = _dsqrt_rn(_dsub_rn(1.0, Bpq)),
      Xi = _ddiv_rn(Bpq, _dadd_rn(Bpqp, Bpqm)),
      Xi_ = -Xi,
      Eta = _ddiv_rn(Bpq, _dmul_rn(_dadd_rn(1.0, Bpqp), _dadd_rn(1.0, Bpqm))),
      Eta_ = -Eta;
    if (Bpq_ < SQRT_HEPS) {
      const double
        Cot2T = _ddiv_rn(E, V),
        Cot2T_ = fabs(Cot2T);
      double TanT;
      if (Cot2T_ >= SQRT_2_HEPS)
        TanT = _dmul_rn(_drcp_rn(Cot2T), 0.5);
      else if (Cot2T_ < SQRT_HEPS)
        TanT = copysign(_drcp_rn(_dadd_rn(Cot2T_, 1.0)), Cot2T);
      else
        TanT = copysign(_drcp_rn(_dadd_rn(Cot2T_, _dsqrt_rn(_fma_rn(Cot2T, Cot2T, 1.0)))), Cot2T);
      if (fabs(TanT) < SQRT_HEPS) {
        CosF = _fma_rn(_dsub_rn(TanT, Eta), Xi, 1.0);
        SinF = _fma_rn(_fma_rn(TanT, Eta, 1.0), Xi_, TanT);
        CosP = _fma_rn(_dadd_rn(TanT, Eta), Xi_, 1.0);
        SinP = _fma_rn(_fma_rn(TanT, Eta_, 1.0), Xi, TanT);
      }
      else {
        const double
          CosT = _drsqrt_rn(_fma_rn(TanT, TanT, 1.0)),
          SinT = _dmul_rn(CosT, TanT);
        CosF = _fma_rn(_fma_rn(CosT, Eta_, SinT), Xi, CosT);
        SinF = _fma_rn(_fma_rn(SinT, Eta, CosT), Xi_, SinT);
        CosP = _fma_rn(_fma_rn(CosT, Eta, SinT), Xi_, CosT);
        SinP = _fma_rn(_fma_rn(SinT, Eta_, CosT), Xi, SinT);
      }
    }
    else {
      const double
        F = _drsqrt_rn(_fma_rn(Bpq, -Bpq, 1.0)),
        Cot2T = _ddiv_rn(E, _dmul_rn(V, F)),
        Cot2T_ = fabs(Cot2T);
      double TanT;
      if (Cot2T_ >= SQRT_2_HEPS)
        TanT = _dmul_rn(_drcp_rn(Cot2T), 0.5);
      else if (Cot2T_ < SQRT_HEPS)
        TanT = copysign(_drcp_rn(_dadd_rn(Cot2T_, 1.0)), Cot2T);
      else
        TanT = copysign(_drcp_rn(_dadd_rn(Cot2T_, _dsqrt_rn(_fma_rn(Cot2T, Cot2T, 1.0)))), Cot2T);
      if (fabs(TanT) < SQRT_HEPS) {
        CosF = _fma_rn(_dsub_rn(TanT, Eta), Xi, 1.0);
        SinF = _fma_rn(_fma_rn(TanT, Eta, 1.0), Xi_, TanT);
        CosP = _fma_rn(_dadd_rn(TanT, Eta), Xi_, 1.0);
        SinP = _fma_rn(_fma_rn(TanT, Eta_, 1.0), Xi, TanT);
      }
      else {
        const double
          CosT = _drsqrt_rn(_fma_rn(TanT, TanT, 1.0)),
          SinT = _dmul_rn(CosT, TanT);
        CosF = _fma_rn(_fma_rn(CosT, Eta_, SinT), Xi, CosT);
        SinF = _fma_rn(_fma_rn(SinT, Eta, CosT), Xi_, SinT);
        CosP = _fma_rn(_fma_rn(CosT, Eta, SinT), Xi_, CosT);
        SinP = _fma_rn(_fma_rn(SinT, Eta_, CosT), Xi, SinT);
      }
      if (F != 1.0) {
        CosF = _dmul_rn(CosF, F);
        SinF = _dmul_rn(SinF, F);
        CosP = _dmul_rn(CosP, F);
        SinP = _dmul_rn(SinP, F);
      }
    }
  }

  // App = _dadd_rn(_dsub_rn(_dmul_rn(_dmul_rn(CosF, CosF), App), _dmul_rn(_dmul_rn(_dmul_rn(CosF, SinP), Apq), 2.0)), _dmul_rn(_dmul_rn(SinP, SinP), Aqq));
  App = _fma_rn(_dmul_rn(SinP, SinP), Aqq, _fma_rn(_dmul_rn(_dmul_rn(CosF, SinP), Apq), -2.0, _dmul_rn(_dmul_rn(CosF, CosF), App)));

  // Aqq = _dadd_rn(_dadd_rn(_dmul_rn(_dmul_rn(SinF, SinF), App), _dmul_rn(_dmul_rn(_dmul_rn(SinF, CosP), Apq), 2.0)), _dmul_rn(_dmul_rn(CosP, CosP), Aqq));
  Aqq = _fma_rn(_dmul_rn(CosP, CosP), Aqq, _fma_rn(_dmul_rn(_dmul_rn(SinF, CosP), Apq),  2.0, _dmul_rn(_dmul_rn(SinF, SinF), App)));
}

#endif /* !DEVICE_CODE_COMMON_ROTATE_HPP */
