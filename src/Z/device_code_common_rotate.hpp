#ifndef DEVICE_CODE_COMMON_ROTATE_HPP
#define DEVICE_CODE_COMMON_ROTATE_HPP

MYDEVFN
#if ((CVG == 4) || (CVG == 5) || (CVG == 6) || (CVG == 7))
int
#else // ((CVG == 0) || (CVG == 1) || (CVG == 2) || (CVG == 3))
void
#endif // ?CVG
zRot
(const cuD App,
 const cuD Aqq,
 const cuD ApqD,
 const cuJ ApqJ,
 const cuD BpqD,
 const cuJ BpqJ,
 const double Bpq_,
 double &CosF,
 cuD &SinFD,
 cuJ &SinFJ,
 double &CosP,
 cuD &_SinPD,
 cuJ &_SinPJ
#if ((CVG == 4) || (CVG == 5) || (CVG == 6) || (CVG == 7))
 , int &fn1
 , int &pn1
#endif // ?CVG
) {
  const double _Bpq_ = -Bpq_;
  const double X = __fma_rn(_Bpq_, Bpq_, 1.0);
  assert(X > 0.0);
  const double E = Aqq - App;
  const double T = __dsqrt_rn(X);
  assert(T > 0.0);
  const double S = copysign(1.0, E);

  cuD _BpqD = BpqD, U;
  cuJ _BpqJ = BpqJ, V;
  if (Bpq_ > 0.0) {
    const double d = __drcp_rn(Bpq_);
    _BpqD *= d;
    _BpqJ *= d;
  }
  else {
    _BpqD = 1.0;
    _BpqJ = 0.0;
  }
  Zmul(U, V, _BpqD, -_BpqJ, ApqD, ApqJ);

  if ((E == 0.0) && (V == 0.0)) {
    const double S1 = my_drsqrt_rn(1.0 + Bpq_);
    const double S2 = my_drsqrt_rn(1.0 - Bpq_);
    const double CG = my_drsqrt_rn(2.0);
    const double SG = -CG;

    CosF = CG * S1;
    _SinPD =  _BpqD * -SG * S1;
    _SinPJ = -_BpqJ * -SG * S1;
     SinFD =  _BpqD *  SG * S2;
     SinFJ =  _BpqJ *  SG * S2;
    CosP = CG * S2;

#if ((CVG == 4) || (CVG == 5) || (CVG == 6) || (CVG == 7))
    fn1 = 1;
    pn1 = 1;
#endif // ?CVG
  }
  else {
    const double TG = scalbn(__ddiv_rn(V, E), 1);
    const double CG = my_drsqrt_rn(__fma_rn(TG, TG, 1.0));
    const double SG = ((CG == 0.0) ? copysign(1.0, TG) : (TG * CG));

    const double T2T = __ddiv_rn(S*__fma_rn(_Bpq_, (App + Aqq), scalbn(U, 1)), T*__dsqrt_rn(__fma_rn(V*V, 4.0, E*E)));
    const double C2T = my_drsqrt_rn(__fma_rn(T2T, T2T, 1.0));
    const double S2T = ((C2T == 0.0) ? copysign(1.0, T2T) : (T2T * C2T));

    const double TC2T = T * C2T;
    const double _T = __drcp_rn(T);
    const double TC2TSG = TC2T * SG;
    const double Yp = __fma_rn(TC2T, CG, __fma_rn(Bpq_, S2T, 1.0));
    const double Ym = __fma_rn(TC2T, CG, __fma_rn(_Bpq_, S2T, 1.0));
    const double _Yp = __drcp_rn(Yp);
    const double _Ym = __drcp_rn(Ym);
    const double CPHI = __dsqrt_rn(scalbn(Yp, -1));
    const double CPSI = __dsqrt_rn(scalbn(Ym, -1));

#if ((CVG == 4) || (CVG == 5) || (CVG == 6) || (CVG == 7))
    fn1 = (CPHI != 1.0);
    pn1 = (CPSI != 1.0);
#endif // ?CVG

    CosF = CPHI * _T;
    CosP = CPSI * _T;

    Zmul(U, V, _BpqD * CPSI, _BpqJ * CPSI, (S2T - Bpq_) * _Ym, TC2TSG * _Ym);
    SinFD = U * _T;
    SinFJ = V * _T;

    Zmul(U, V, _BpqD * CPHI, -_BpqJ * CPHI, (S2T + Bpq_) * _Yp, -TC2TSG * _Yp);
    _SinPD = -U * _T;
    _SinPJ = -V * _T;
  }
#if ((CVG == 4) || (CVG == 5) || (CVG == 6) || (CVG == 7))
  return (fn1 || pn1);
#endif // ?CVG
}

#endif // !DEVICE_CODE_COMMON_ROTATE_HPP
