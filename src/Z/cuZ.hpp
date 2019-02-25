#ifndef CU_Z_HPP
#define CU_Z_HPP

MYDEVFN void Zfma(cuD &dD, cuJ &dJ, const cuD aD, const cuJ aJ, const cuD bD, const cuJ bJ, const cuD cD, const cuJ cJ)
{
  /*
    a * b + c =
    aD*bD-aJ*bJ+cD + i*(aD*bJ+aJ*bD+cJ) =
    fma(aD,bD,fma(-aJ,bJ,cD)) + i*fma(aD,bJ,fma(aJ,bD,cJ))
   */
  const cuD x = __fma_rn(aD, bD, __fma_rn(-aJ, bJ, cD));
  const cuJ y = __fma_rn(aD, bJ, __fma_rn( aJ, bD, cJ));
  dD = x;
  dJ = y;
}

MYDEVFN void Zmul(cuD &cD, cuJ &cJ, const cuD aD, const cuJ aJ, const cuD bD, const cuJ bJ)
{
  const cuD x = __fma_rn(aD, bD, -aJ * bJ);
  const cuJ y = __fma_rn(aD, bJ,  aJ * bD);
  cD = x;
  cJ = y;
}

#endif /* !CU_Z_HPP */
