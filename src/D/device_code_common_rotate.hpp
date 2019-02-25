#ifndef DEVICE_CODE_COMMON_ROTATE_HPP
#define DEVICE_CODE_COMMON_ROTATE_HPP

MYDEVFN
#if ((CVG == 4) || (CVG == 5) || (CVG == 6) || (CVG == 7))
int
#else // ((CVG == 0) || (CVG == 1) || (CVG == 2) || (CVG == 3))
void
#endif // ?CVG
dRot
(double &App,
 double &Aqq,
 const double Apq,
 const double Bpq,
 const double Bpq_,
 double &CosF,
 double &SinF,
 double &CosP,
 double &SinP
#if ((CVG == 4) || (CVG == 5) || (CVG == 6) || (CVG == 7))
 , int &fn1
 , int &pn1
#endif // ?CVG
 ) {
  const double
    E = (Aqq - App),
    V = __fma_rn(Bpq, -(App + Aqq), scalbn(Apq, 1));

  if ((V == 0.0) && (E == 0.0)) {
    const double S1 = my_drsqrt_rn(1.0 + Bpq_);
    const double S2 = my_drsqrt_rn(1.0 - Bpq_);
    const double CG = my_drsqrt_rn(2.0);
    const double SG = -CG;

    CosF = CG * S1;
    SinP = SG * S1;
    SinF = SG * S2;
    CosP = CG * S2;

#if ((CVG == 4) || (CVG == 5) || (CVG == 6) || (CVG == 7))
    fn1 = 1;
    pn1 = 1;
#endif // ?CVG
  }
  else {
    const double
      Bpqp = __dsqrt_rn(1.0 + Bpq),
      Bpqm = __dsqrt_rn(1.0 - Bpq),
      Xi = __ddiv_rn(Bpq, (Bpqp + Bpqm)),
      Xi_ = -Xi,
      Eta = __ddiv_rn(Bpq, ((1.0 + Bpqp) * (1.0 + Bpqm))),
      Eta_ = -Eta;
    if (Bpq_ < SQRT_HEPS) {
      const double
        Cot2T = __ddiv_rn(E, V),
        Cot2T_ = fabs(Cot2T);
      double TanT;
      if (Cot2T_ >= SQRT_2_HEPS)
        TanT = scalbn(__drcp_rn(Cot2T), -1);
      else if (Cot2T_ < SQRT_HEPS)
        TanT = copysign(__drcp_rn(Cot2T_ + 1.0), Cot2T);
      else
        TanT = copysign(__drcp_rn(Cot2T_ + __dsqrt_rn(__fma_rn(Cot2T, Cot2T, 1.0))), Cot2T);
      if (fabs(TanT) < SQRT_HEPS) {
        CosF = __fma_rn(TanT - Eta, Xi, 1.0);
        SinF = __fma_rn(__fma_rn(TanT, Eta, 1.0), Xi_, TanT);
        CosP = __fma_rn(TanT + Eta, Xi_, 1.0);
        SinP = __fma_rn(__fma_rn(TanT, Eta_, 1.0), Xi, TanT);
      }
      else {
        const double
          CosT = my_drsqrt_rn(__fma_rn(TanT, TanT, 1.0)),
          SinT = CosT * TanT;
        CosF = __fma_rn(__fma_rn(CosT, Eta_, SinT), Xi, CosT);
        SinF = __fma_rn(__fma_rn(SinT, Eta, CosT), Xi_, SinT);
        CosP = __fma_rn(__fma_rn(CosT, Eta, SinT), Xi_, CosT);
        SinP = __fma_rn(__fma_rn(SinT, Eta_, CosT), Xi, SinT);
      }
#if ((CVG == 4) || (CVG == 5) || (CVG == 6) || (CVG == 7))
      fn1 = (CosF != 1.0);
      pn1 = (CosP != 1.0);
#endif // ?CVG
    }
    else {
      const double
        F = my_drsqrt_rn(__fma_rn(Bpq, -Bpq, 1.0)),
        Cot2T = __ddiv_rn(E, V * F),
        Cot2T_ = fabs(Cot2T);
      double TanT;
      if (Cot2T_ >= SQRT_2_HEPS)
        TanT = scalbn(__drcp_rn(Cot2T), -1);
      else if (Cot2T_ < SQRT_HEPS)
        TanT = copysign(__drcp_rn(Cot2T_ + 1.0), Cot2T);
      else
        TanT = copysign(__drcp_rn(Cot2T_ + __dsqrt_rn(__fma_rn(Cot2T, Cot2T, 1.0))), Cot2T);
      if (fabs(TanT) < SQRT_HEPS) {
        CosF = __fma_rn(TanT - Eta, Xi, 1.0);
        SinF = __fma_rn(__fma_rn(TanT, Eta, 1.0), Xi_, TanT);
        CosP = __fma_rn(TanT + Eta, Xi_, 1.0);
        SinP = __fma_rn(__fma_rn(TanT, Eta_, 1.0), Xi, TanT);
      }
      else {
        const double
          CosT = my_drsqrt_rn(__fma_rn(TanT, TanT, 1.0)),
          SinT = CosT * TanT;
        CosF = __fma_rn(__fma_rn(CosT, Eta_, SinT), Xi, CosT);
        SinF = __fma_rn(__fma_rn(SinT, Eta, CosT), Xi_, SinT);
        CosP = __fma_rn(__fma_rn(CosT, Eta, SinT), Xi_, CosT);
        SinP = __fma_rn(__fma_rn(SinT, Eta_, CosT), Xi, SinT);
      }
#if ((CVG == 4) || (CVG == 5) || (CVG == 6) || (CVG == 7))
      fn1 = (CosF != 1.0);
      pn1 = (CosP != 1.0);
#endif // ?CVG
      if (F != 1.0) {
        CosF *= F;
        SinF *= F;
        CosP *= F;
        SinP *= F;
      }
    }
  }

  App = CosF*CosF*App - scalbn(CosF*SinP*Apq, 1) + SinP*SinP*Aqq;
  Aqq = SinF*SinF*App + scalbn(SinF*CosP*Apq, 1) + CosP*CosP*Aqq;

#if ((CVG == 4) || (CVG == 5) || (CVG == 6) || (CVG == 7))
  return (fn1 || pn1);
#endif // ?CVG
}

#endif // !DEVICE_CODE_COMMON_ROTATE_HPP
