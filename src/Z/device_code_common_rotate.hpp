#ifndef DEVICE_CODE_COMMON_ROTATE_HPP
#define DEVICE_CODE_COMMON_ROTATE_HPP

MYDEVFN
#if ((CVG == 4) || (CVG == 5) || (CVG == 6) || (CVG == 7))
int
#else /* ((CVG == 0) || (CVG == 1) || (CVG == 2) || (CVG == 3)) */
void
#endif /* ?CVG */
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
#endif /* ?CVG */
) {
  const double _Bpq_ = -Bpq_;
  const double X = _fma_rn(_Bpq_, Bpq_, 1.0);
  assert(X > 0.0);
  const double E = _dsub_rn(Aqq, App);
  const double T = _dsqrt_rn(X);
  assert(T > 0.0);
  const double S = copysign(1.0, E);

  cuD _BpqD = BpqD, U;
  cuJ _BpqJ = BpqJ, V;
  if (Bpq_ > 0.0) {
    const double d = _drcp_rn(Bpq_);
    _BpqD = _dmul_rn(_BpqD, d);
    _BpqJ = _dmul_rn(_BpqJ, d);
  }
  else {
    _BpqD = 1.0;
    _BpqJ = 0.0;
  }
  Zmul(U, V, _BpqD, -_BpqJ, ApqD, ApqJ);

  if ((E == 0.0) && (V == 0.0)) {
    const double S1 = _drsqrt_rn(_dadd_rn(1.0, Bpq_));
    const double S2 = _drsqrt_rn(_dsub_rn(1.0, Bpq_));
    const double CG = RSQRT_2;
    const double SG = -CG;

    CosF = _dmul_rn(CG, S1);
    _SinPD = _dmul_rn(_dmul_rn(_BpqD, -SG), S1);
    _SinPJ = _dmul_rn(_dmul_rn(_BpqJ, SG), S1);
    SinFD = _dmul_rn(_dmul_rn(_BpqD, SG), S2);
    SinFJ = _dmul_rn(_dmul_rn(_BpqJ, SG), S2);
    CosP = _dmul_rn(CG, S2);

#if ((CVG == 4) || (CVG == 5) || (CVG == 6) || (CVG == 7))
    fn1 = 1;
    pn1 = 1;
#endif /* ?CVG */
  }
  else {
    const double TG = _dmul_rn(_ddiv_rn(V, E), 2.0);
    const double CG = _drsqrt_rn(_fma_rn(TG, TG, 1.0));
    const double SG = ((CG == 0.0) ? copysign(1.0, TG) : _dmul_rn(TG, CG));

    const double T2T = _ddiv_rn(_dmul_rn(S, _fma_rn(_Bpq_, _dadd_rn(App, Aqq), _dmul_rn(U, 2.0))), _dmul_rn(T, _dsqrt_rn(_fma_rn(_dmul_rn(V, V), 4.0, _dmul_rn(E, E)))));
    const double C2T = _drsqrt_rn(_fma_rn(T2T, T2T, 1.0));
    const double S2T = ((C2T == 0.0) ? copysign(1.0, T2T) : _dmul_rn(T2T, C2T));

    const double TC2T = _dmul_rn(T, C2T);
    const double _T = _drcp_rn(T);
    const double TC2TSG = _dmul_rn(TC2T, SG);
    const double Yp = _fma_rn(TC2T, CG, _fma_rn(Bpq_, S2T, 1.0));
    const double Ym = _fma_rn(TC2T, CG, _fma_rn(_Bpq_, S2T, 1.0));
    const double _Yp = _drcp_rn(Yp);
    const double _Ym = _drcp_rn(Ym);
    const double CPHI = _dsqrt_rn(_dmul_rn(Yp, 0.5));
    const double CPSI = _dsqrt_rn(_dmul_rn(Ym, 0.5));

#if ((CVG == 4) || (CVG == 5) || (CVG == 6) || (CVG == 7))
    fn1 = (CPHI != 1.0);
    pn1 = (CPSI != 1.0);
#endif /* ?CVG */

    CosF = _dmul_rn(CPHI, _T);
    CosP = _dmul_rn(CPSI, _T);

    Zmul(U, V, _dmul_rn(_BpqD, CPSI), _dmul_rn(_BpqJ, CPSI), _dmul_rn(_dsub_rn(S2T, Bpq_), _Ym), _dmul_rn(TC2TSG, _Ym));
    SinFD = _dmul_rn(U, _T);
    SinFJ = _dmul_rn(V, _T);

    Zmul(U, V, _dmul_rn(_BpqD, CPHI), -_dmul_rn(_BpqJ, CPHI), _dmul_rn(_dadd_rn(S2T, Bpq_), _Yp), -_dmul_rn(TC2TSG, _Yp));
    _SinPD = -_dmul_rn(U, _T);
    _SinPJ = -_dmul_rn(V, _T);
  }
#if ((CVG == 4) || (CVG == 5) || (CVG == 6) || (CVG == 7))
  return (fn1 || pn1);
#endif /* ?CVG */
}

#endif /* !DEVICE_CODE_COMMON_ROTATE_HPP */
