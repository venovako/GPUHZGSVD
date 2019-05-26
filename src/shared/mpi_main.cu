// mpi_main.cu: test driver.

#include "HZ.hpp"
#include "HZ_L.hpp"
#include "HZ_L3.hpp"

#include "my_utils.hpp"
#include "cuda_memory_helper.hpp"

int main(int argc, char *argv[])
{
  if (9 != argc) {
    DIE("Arguments: SNP0 SNP1 SNP2 ALG MF MG N FN");
  }

  const char *const ca_exe = argv[0];
  const char *const ca_snp0 = argv[1];
  const char *const ca_snp1 = argv[2];
  const char *const ca_snp2 = argv[3];
  const char *const ca_alg = argv[4];
  const char *const ca_mF = argv[5];
  const char *const ca_mG = argv[6];
  const char *const ca_n = argv[7];
  const char *const ca_fn = argv[8];

  const unsigned routine = static_cast<unsigned>(atou(ca_alg));

  const size_t mF = atou(ca_mF);
  const size_t mG = atou(ca_mG);
  const size_t n = atou(ca_n);
  if (n > mF) {
    DIE("N > MF");
  }
  if (n > mG) {
    DIE("N > MG");
  }

  const unsigned snp0 = static_cast<unsigned>(atou(ca_snp0));
  if ((snp0 != STRAT_CYCWOR) && (snp0 != STRAT_MMSTEP)) {
    DIE("SNP0 \\notin { 2, 4 }");
  }
  const unsigned snp1 = static_cast<unsigned>(atou(ca_snp1));
  if ((snp1 != STRAT_CYCWOR) && (snp1 != STRAT_MMSTEP)) {
    DIE("SNP1 \\notin { 2, 4 }");
  }
  const unsigned snp2 = static_cast<unsigned>(atou(ca_snp2));
  if ((snp2 != (STRAT_CYCWOR + 1u)) && (snp2 != (STRAT_MMSTEP + 1u))) {
    DIE("SNP2 \\notin { 3, 5 }");
  }
  if (!*ca_fn) {
    DIE("invalid argument FN");
  }

  if (init_MPI(&argc, &argv)) {
    (void)snprintf(err_msg, err_msg_size, "%s[%d] init_MPI failed\n", ca_exe, mpi_rank);
    DIE(err_msg);
  }
  const size_t gpu = static_cast<size_t>(mpi_rank);
  if (!mpi_cuda_aware) {
    DIE("MPI is not CUDA aware");
  }
  const size_t gpus = static_cast<size_t>(mpi_size);
  if (gpus < 2u) {
    DIE("MPI_COMM_WORLD size < 2");
  }
  const size_t n2 = (gpus << 1u);
  if (n < n2) {
    DIE("N < n2");
  }

  const int dev = assign_dev2host();
  if (dev < 0) {
    DIE("assign_dev2host failed");
  }

  const int dcc = configureGPU(dev);
#ifndef NDEBUG
  (void)fprintf(stdout, "[%u] device(%d) has CC(%d)\n", gpu, dev, dcc);
  (void)fflush(stdout);
#endif // !NDEBUG

  size_t mF_ = 0u, mG_ = 0u, n_ = 0u;
  border_sizes(gpus, mF, mG, n, mF_, mG_, n_);
  const size_t n_gpu = n_ / gpus;
  const size_t n_col = (n_gpu >> 1u);

  const size_t n0 = (HZ_L1_NCOLB << 1u);
  const size_t n1 = n_gpu / HZ_L1_NCOLB;
  if (n_gpu % HZ_L1_NCOLB) {
    DIE("n_gpu % 16 != 0");
  }
  init_strats(snp0, n0, snp1, n1, snp2, n2);

  const size_t p = static_cast<size_t>(strat2[0u][gpu][0u][0u]);
  const size_t q = static_cast<size_t>(strat2[0u][gpu][0u][1u]);

  size_t
    ldhF = mF_,
    ldhG = mG_,
    ldhV = n_;

  const size_t p_ = p * n_col;
  const size_t n_p = ((p_ >= n) ? static_cast<size_t>(0u) : (((p_ + n_col) > n) ? (n - p_) : n_col));
  const MPI_Offset opF = static_cast<MPI_Offset>((p_ * mF)
#ifdef USE_COMPLEX
    * 2u
#endif // USE_COMPLEX
  );
  const MPI_Offset opG = static_cast<MPI_Offset>((p_ * mG)
#ifdef USE_COMPLEX
    * 2u
#endif // USE_COMPLEX
  );
  const MPI_Offset opV = static_cast<MPI_Offset>((p_ * n)
#ifdef USE_COMPLEX
    * 2u
#endif // USE_COMPLEX
  );
  const MPI_Offset opS = static_cast<MPI_Offset>(p_);

  const size_t q_ = q * n_col; 
  const size_t n_q = ((q_ >= n) ? static_cast<size_t>(0u) : (((q_ + n_col) > n) ? (n - q_) : n_col));
  const MPI_Offset oqF = static_cast<MPI_Offset>((q_ * mF)
#ifdef USE_COMPLEX
    * 2u
#endif // USE_COMPLEX
  );
  const MPI_Offset oqG = static_cast<MPI_Offset>((q_ * mG)
#ifdef USE_COMPLEX
    * 2u
#endif // USE_COMPLEX
  );
  const MPI_Offset oqV = static_cast<MPI_Offset>((q_ * n)
#ifdef USE_COMPLEX
    * 2u
#endif // USE_COMPLEX
  );
  const MPI_Offset oqS = static_cast<MPI_Offset>(q_);

  char *const fn = static_cast<char*>(calloc(strlen(ca_fn) + 4u, sizeof(char)));
  SYSP_CALL(fn);
#ifdef USE_COMPLEX
  double *const buf = static_cast<double*>(calloc((((mF >= mG) ? mF : mG) * 2u), sizeof(double)));
  SYSP_CALL(buf);
#endif // USE_COMPLEX

  MPI_File fh = MPI_FILE_NULL;
  size_t ldA = static_cast<size_t>(0u);

  ldA = ldhF;
#ifdef USE_COMPLEX
  cuD *const hFD = allocHostMtx<cuD>(ldA, mF_, n_gpu, true);
  SYSP_CALL(hFD);
  cuJ *const hFJ = allocHostMtx<cuJ>(ldA, mF_, n_gpu, true);
  SYSP_CALL(hFJ);
#else // !USE_COMPLEX
  double *const hF = allocHostMtx<double>(ldA, mF_, n_gpu, true);
  SYSP_CALL(hF);
#endif // ?USE_COMPLEX
  ldhF = ldA;

  if (MPI_File_open(MPI_COMM_WORLD, strcat(strcpy(fn, ca_fn), ".Y"), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh)) {
    DIE("MPI_File_open(Y)");
  }
  if (MPI_File_set_view(fh, 0, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL)) {
    DIE("MPI_File_set_view(Y)");
  }
#ifdef USE_COMPLEX
  for (size_t j = 0u; j < n_p; ++j) {
    if (MPI_File_read_at(fh, (opF + j * mF * 2u), buf, (mF * 2u), MPI_DOUBLE, MPI_STATUS_IGNORE)) {
      DIE("MPI_File_read_at(Y)p");
    }
    const size_t o = (ldA * j);
    cuD *const cD = (hFD + o);
    cuJ *const cJ = (hFJ + o);
    for (size_t i = 0u; i < mF; ++i) {
      const size_t i2 = (i * 2u);
      cD[i] = static_cast<cuD>(buf[i2]);
      cJ[i] = static_cast<cuJ>(buf[i2 + 1u]);
    }
  }
  for (size_t j = 0u; j < n_q; ++j) {
    if (MPI_File_read_at(fh, (oqF + j * mF * 2u), buf, (mF * 2u), MPI_DOUBLE, MPI_STATUS_IGNORE)) {
      DIE("MPI_File_read_at(Y)q");
    }
    const size_t o = (ldA * (n_col + j));
    cuD *const cD = (hFD + o);
    cuJ *const cJ = (hFJ + o);
    for (size_t i = 0u; i < mF; ++i) {
      const size_t i2 = (i * 2u);
      cD[i] = static_cast<cuD>(buf[i2]);
      cJ[i] = static_cast<cuJ>(buf[i2 + 1u]);
    }
  }
#else // !USE_COMPLEX
  for (size_t j = 0u; j < n_p; ++j) {
    if (MPI_File_read_at(fh, (opF + j * mF), (hF + ldA * j), mF, MPI_DOUBLE, MPI_STATUS_IGNORE)) {
      DIE("MPI_File_read_at(Y)p");
    }
  }
  for (size_t j = 0u; j < n_q; ++j) {
    if (MPI_File_read_at(fh, (oqF + j * mF), (hF + ldA * (n_col + j)), mF, MPI_DOUBLE, MPI_STATUS_IGNORE)) {
      DIE("MPI_File_read_at(Y)q");
    }
  }
#endif // ?USE_COMPLEX
  if (MPI_File_close(&fh)) {
    DIE("MPI_File_close(Y)");
  }

  if ((p_ + n_col) > n) {
    if (p_ >= n) {
      const size_t o = (p_ - n);
#ifdef USE_COMPLEX
      SYSI_CALL(bdinit((mF + o), (o + 1u), hFD, ldA));
#else // !USE_COMPLEX
      SYSI_CALL(bdinit((mF + o), (o + 1u), hF, ldA));
#endif // ?USE_COMPLEX
    }
    else {
      const size_t f = (n - p_);
#ifdef USE_COMPLEX
      SYSI_CALL(bdinit(mF, (n_col - f), (hFD + ldA * f), ldA));
#else // !USE_COMPLEX
      SYSI_CALL(bdinit(mF, (n_col - f), (hF + ldA * f), ldA));
#endif // ?USE_COMPLEX
    }
  }
  if ((q_ + n_col) > n) {
    if (q_ >= n) {
      const size_t o = (q_ - n);
#ifdef USE_COMPLEX
      SYSI_CALL(bdinit((mF + o), (o + 1u), (hFD + ldA * n_col), ldA));
#else // !USE_COMPLEX
      SYSI_CALL(bdinit((mF + o), (o + 1u), (hF + ldA * n_col), ldA));
#endif // ?USE_COMPLEX
    }
    else {
      const size_t f = (n - p_);
#ifdef USE_COMPLEX
      SYSI_CALL(bdinit(mF, (n_col - f), (hFD + ldA * (n_col + f)), ldA));
#else // !USE_COMPLEX
      SYSI_CALL(bdinit(mF, (n_col - f), (hF + ldA * (n_col + f)), ldA));
#endif // ?USE_COMPLEX
    }
  }

  ldA = ldhG;
#ifdef USE_COMPLEX
  cuD *const hGD = allocHostMtx<cuD>(ldA, mG_, n_gpu, true);
  SYSP_CALL(hGD);
  cuJ *const hGJ = allocHostMtx<cuJ>(ldA, mG_, n_gpu, true);
  SYSP_CALL(hGJ);
#else // !USE_COMPLEX
  double *const hG = allocHostMtx<double>(ldA, mG_, n_gpu, true);
  SYSP_CALL(hG);
#endif // ?USE_COMPLEX
  ldhG = ldA;

  if (MPI_File_open(MPI_COMM_WORLD, strcat(strcpy(fn, ca_fn), ".W"), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh)) {
    DIE("MPI_File_open(W)");
  }
  if (MPI_File_set_view(fh, 0, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL)) {
    DIE("MPI_File_set_view(W)");
  }
#ifdef USE_COMPLEX
  for (size_t j = 0u; j < n_p; ++j) {
    if (MPI_File_read_at(fh, (opG + j * mG * 2u), buf, (mG * 2u), MPI_DOUBLE, MPI_STATUS_IGNORE)) {
      DIE("MPI_File_read_at(W)p");
    }
    const size_t o = (ldA * j);
    cuD *const cD = (hGD + o);
    cuJ *const cJ = (hGJ + o);
    for (size_t i = 0u; i < mG; ++i) {
      const size_t i2 = (i * 2u);
      cD[i] = static_cast<cuD>(buf[i2]);
      cJ[i] = static_cast<cuJ>(buf[i2 + 1u]);
    }
  }
  for (size_t j = 0u; j < n_q; ++j) {
    if (MPI_File_read_at(fh, (oqG + j * mG * 2u), buf, (mG * 2u), MPI_DOUBLE, MPI_STATUS_IGNORE)) {
      DIE("MPI_File_read_at(W)q");
    }
    const size_t o = (ldA * (n_col + j));
    cuD *const cD = (hGD + o);
    cuJ *const cJ = (hGJ + o);
    for (size_t i = 0u; i < mG; ++i) {
      const size_t i2 = (i * 2u);
      cD[i] = static_cast<cuD>(buf[i2]);
      cJ[i] = static_cast<cuJ>(buf[i2 + 1u]);
    }
  }
#else // !USE_COMPLEX
  for (size_t j = 0u; j < n_p; ++j) {
    if (MPI_File_read_at(fh, (opG + j * mG), (hG + ldA * j), mG, MPI_DOUBLE, MPI_STATUS_IGNORE)) {
      DIE("MPI_File_read_at(W)p");
    }
  }
  for (size_t j = 0u; j < n_q; ++j) {
    if (MPI_File_read_at(fh, (oqG + j * mG), (hG + ldA * (n_col + j)), mG, MPI_DOUBLE, MPI_STATUS_IGNORE)) {
      DIE("MPI_File_read_at(W)q");
    }
  }
#endif // ?USE_COMPLEX
  if (MPI_File_close(&fh)) {
    DIE("MPI_File_close(W)");
  }

  if ((p_ + n_col) > n) {
    if (p_ >= n) {
      const size_t o = (p_ - n);
#ifdef USE_COMPLEX
      SYSI_CALL(bdinit((mG + o), (o + 1u), hGD, ldA));
#else // !USE_COMPLEX
      SYSI_CALL(bdinit((mG + o), (o + 1u), hG, ldA));
#endif // ?USE_COMPLEX
    }
    else {
      const size_t f = (n - p_);
#ifdef USE_COMPLEX
      SYSI_CALL(bdinit(mG, (n_col - f), (hGD + ldA * f), ldA));
#else // !USE_COMPLEX
      SYSI_CALL(bdinit(mG, (n_col - f), (hG + ldA * f), ldA));
#endif // ?USE_COMPLEX
    }
  }
  if ((q_ + n_col) > n) {
    if (q_ >= n) {
      const size_t o = (q_ - n);
#ifdef USE_COMPLEX
      SYSI_CALL(bdinit((mG + o), (o + 1u), (hGD + ldA * n_col), ldA));
#else // !USE_COMPLEX
      SYSI_CALL(bdinit((mG + o), (o + 1u), (hG + ldA * n_col), ldA));
#endif // ?USE_COMPLEX
    }
    else {
      const size_t f = (n - p_);
#ifdef USE_COMPLEX
      SYSI_CALL(bdinit(mG, (n_col - f), (hGD + ldA * (n_col + f)), ldA));
#else // !USE_COMPLEX
      SYSI_CALL(bdinit(mG, (n_col - f), (hG + ldA * (n_col + f)), ldA));
#endif // ?USE_COMPLEX
    }
  }

  ldA = ldhV;
#ifdef USE_COMPLEX
  cuD *const hVD = allocHostMtx<cuD>(ldA, n_gpu, n_gpu, true);
  SYSP_CALL(hVD);
  cuJ *const hVJ = allocHostMtx<cuJ>(ldA, n_gpu, n_gpu, true);
  SYSP_CALL(hVJ);
#else // !USE_COMPLEX
  double *const hV = allocHostMtx<double>(ldA, n_gpu, n_gpu, true);
  SYSP_CALL(hV);
#endif // ?USE_COMPLEX
  ldhV = ldA;

  double *const hS = allocHostVec<double>(n_gpu);
  SYSP_CALL(hS);
  double *const hH = allocHostVec<double>(n_gpu);
  SYSP_CALL(hH);
  double *const hK = allocHostVec<double>(n_gpu);
  SYSP_CALL(hK);

  unsigned glbSwp = 0u;
  unsigned long long glb_s = 0ull, glb_b = 0ull;
  double timing[4u] = { -0.0, -0.0, -0.0, -0.0 };

#ifdef USE_COMPLEX
  const int ret = HZ_L3(routine, gpu, gpus, mF_, mG_, n_, n_gpu, n_col, hFD, hFJ, ldhF, hGD, hGJ, ldhG, hVD, hVJ, ldhV, hS, hH, hK, glbSwp, glb_s, glb_b, timing);
#else // !USE_COMPLEX
  const int ret = HZ_L3(routine, gpu, gpus, mF_, mG_, n_, n_gpu, n_col, hF, ldhF, hG, ldhG, hV, ldhV, hS, hH, hK, glbSwp, glb_s, glb_b, timing);
#endif // ?USE_COMPLEX

  if (ret) {
    (void)snprintf(err_msg, err_msg_size, "%s: error %d", ca_exe, ret);
    DIE(err_msg);
  }
  else if (!gpu) {
    (void)fprintf(stdout, "GLB_ROT_S(%20llu), GLB_ROT_B(%20llu)\n", glb_s, glb_b);
    (void)fflush(stdout);
    (void)fprintf(stdout, "%#12.6f s %2u sweeps\n", *timing, glbSwp);
    (void)fflush(stdout);
  }

  if (MPI_File_open(MPI_COMM_WORLD, strcat(strcpy(fn, ca_fn), ".YU"), (MPI_MODE_WRONLY | MPI_MODE_CREATE), MPI_INFO_NULL, &fh)) {
    DIE("MPI_File_open(YU)");
  }
  if (MPI_File_set_size(fh, (mF * n * sizeof(double)
#ifdef USE_COMPLEX
    * 2u
#endif // USE_COMPLEX
  ))) {
    DIE("MPI_File_set_size(YU)");
  }
  if (MPI_File_sync(fh)) {
    DIE("MPI_File_sync(YU)");
  }
  if (MPI_File_set_view(fh, 0, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL)) {
    DIE("MPI_File_set_view(YU)");
  }
#ifdef USE_COMPLEX
  for (size_t j = 0u; j < n_p; ++j) {
    const size_t o = (ldhF * j);
    const cuD *const cD = (hFD + o);
    const cuJ *const cJ = (hFJ + o);
    for (size_t i = 0u; i < mF; ++i) {
      const size_t i2 = (i * 2u);
      buf[i2] = static_cast<double>(cD[i]);
      buf[i2 + 1u] = static_cast<double>(cJ[i]);
    }
    if (MPI_File_write_at(fh, (opF + j * mF * 2u), buf, (mF * 2u), MPI_DOUBLE, MPI_STATUS_IGNORE)) {
      DIE("MPI_File_write_at(YU)p");
    }
  }
  if (MPI_File_sync(fh)) {
    DIE("MPI_File_sync(YU)p");
  }
  for (size_t j = 0u; j < n_q; ++j) {
    const size_t o = (ldhF * (n_col + j));
    const cuD *const cD = (hFD + o);
    const cuJ *const cJ = (hFJ + o);
    for (size_t i = 0u; i < mF; ++i) {
      const size_t i2 = (i * 2u);
      buf[i2] = static_cast<double>(cD[i]);
      buf[i2 + 1u] = static_cast<double>(cJ[i]);
    }
    if (MPI_File_write_at(fh, (oqF + j * mF * 2u), buf, (mF * 2u), MPI_DOUBLE, MPI_STATUS_IGNORE)) {
      DIE("MPI_File_write_at(YU)q");
    }
  }
  if (MPI_File_sync(fh)) {
    DIE("MPI_File_sync(YU)q");
  }
#else // !USE_COMPLEX
  for (size_t j = 0u; j < n_p; ++j) {
    if (MPI_File_write_at(fh, (opF + j * mF), (hF + ldhF * j), mF, MPI_DOUBLE, MPI_STATUS_IGNORE)) {
      DIE("MPI_File_write_at(YU)p");
    }
  }
  if (MPI_File_sync(fh)) {
    DIE("MPI_File_sync(YU)p");
  }
  for (size_t j = 0u; j < n_q; ++j) {
    if (MPI_File_write_at(fh, (oqF + j * mF), (hF + ldhF * (n_col + j)), mF, MPI_DOUBLE, MPI_STATUS_IGNORE)) {
      DIE("MPI_File_write_at(YU)q");
    }
  }
  if (MPI_File_sync(fh)) {
    DIE("MPI_File_sync(YU)q");
  }
#endif // ?USE_COMPLEX
  if (MPI_File_close(&fh)) {
    DIE("MPI_File_close(YU)");
  }

  if (MPI_File_open(MPI_COMM_WORLD, strcat(strcpy(fn, ca_fn), ".WV"), (MPI_MODE_WRONLY | MPI_MODE_CREATE), MPI_INFO_NULL, &fh)) {
    DIE("MPI_File_open(WV)");
  }
  if (MPI_File_set_size(fh, (mG * n * sizeof(double)
#ifdef USE_COMPLEX
    * 2u
#endif // USE_COMPLEX
  ))) {
    DIE("MPI_File_set_size(WV)");
  }
  if (MPI_File_sync(fh)) {
    DIE("MPI_File_sync(WV)");
  }
  if (MPI_File_set_view(fh, 0, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL)) {
    DIE("MPI_File_set_view(WV)");
  }
#ifdef USE_COMPLEX
  for (size_t j = 0u; j < n_p; ++j) {
    const size_t o = (ldhG * j);
    const cuD *const cD = (hGD + o);
    const cuJ *const cJ = (hGJ + o);
    for (size_t i = 0u; i < mG; ++i) {
      const size_t i2 = (i * 2u);
      buf[i2] = static_cast<double>(cD[i]);
      buf[i2 + 1u] = static_cast<double>(cJ[i]);
    }
    if (MPI_File_write_at(fh, (opG + j * mG * 2u), buf, (mG * 2u), MPI_DOUBLE, MPI_STATUS_IGNORE)) {
      DIE("MPI_File_write_at(WV)p");
    }
  }
  if (MPI_File_sync(fh)) {
    DIE("MPI_File_sync(WV)p");
  }
  for (size_t j = 0u; j < n_q; ++j) {
    const size_t o = (ldhG * (n_col + j));
    const cuD *const cD = (hGD + o);
    const cuJ *const cJ = (hGJ + o);
    for (size_t i = 0u; i < mG; ++i) {
      const size_t i2 = (i * 2u);
      buf[i2] = static_cast<double>(cD[i]);
      buf[i2 + 1u] = static_cast<double>(cJ[i]);
    }
    if (MPI_File_write_at(fh, (oqG + j * mG * 2u), buf, (mG * 2u), MPI_DOUBLE, MPI_STATUS_IGNORE)) {
      DIE("MPI_File_write_at(WV)q");
    }
  }
  if (MPI_File_sync(fh)) {
    DIE("MPI_File_sync(WV)q");
  }
#else // !USE_COMPLEX
  for (size_t j = 0u; j < n_p; ++j) {
    if (MPI_File_write_at(fh, (opG + j * mG), (hG + ldhG * j), mG, MPI_DOUBLE, MPI_STATUS_IGNORE)) {
      DIE("MPI_File_write_at(WV)p");
    }
  }
  if (MPI_File_sync(fh)) {
    DIE("MPI_File_sync(WV)p");
  }
  for (size_t j = 0u; j < n_q; ++j) {
    if (MPI_File_write_at(fh, (oqG + j * mG), (hG + ldhG * (n_col + j)), mG, MPI_DOUBLE, MPI_STATUS_IGNORE)) {
      DIE("MPI_File_write_at(WV)q");
    }
  }
  if (MPI_File_sync(fh)) {
    DIE("MPI_File_sync(WV)q");
  }
#endif // ?USE_COMPLEX
  if (MPI_File_close(&fh)) {
    DIE("MPI_File_close(WV)");
  }

  if (MPI_File_open(MPI_COMM_WORLD, strcat(strcpy(fn, ca_fn), ".Z"), (MPI_MODE_WRONLY | MPI_MODE_CREATE), MPI_INFO_NULL, &fh)) {
    DIE("MPI_File_open(Z)");
  }
  if (MPI_File_set_size(fh, (n * n * sizeof(double)
#ifdef USE_COMPLEX
    * 2u
#endif // USE_COMPLEX
  ))) {
    DIE("MPI_File_set_size(Z)");
  }
  if (MPI_File_sync(fh)) {
    DIE("MPI_File_sync(Z)");
  }
  if (MPI_File_set_view(fh, 0, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL)) {
    DIE("MPI_File_set_view(Z)");
  }
#ifdef USE_COMPLEX
  for (size_t j = 0u; j < n_p; ++j) {
    const size_t o = (ldhV * j);
    const cuD *const cD = (hVD + o);
    const cuJ *const cJ = (hVJ + o);
    for (size_t i = 0u; i < n; ++i) {
      const size_t i2 = (i * 2u);
      buf[i2] = static_cast<double>(cD[i]);
      buf[i2 + 1u] = static_cast<double>(cJ[i]);
    }
    if (MPI_File_write_at(fh, (opV + j * n * 2u), buf, (n * 2u), MPI_DOUBLE, MPI_STATUS_IGNORE)) {
      DIE("MPI_File_write_at(Z)p");
    }
  }
  if (MPI_File_sync(fh)) {
    DIE("MPI_File_sync(Z)p");
  }
  for (size_t j = 0u; j < n_q; ++j) {
    const size_t o = (ldhV * (n_col + j));
    const cuD *const cD = (hVD + o);
    const cuJ *const cJ = (hVJ + o);
    for (size_t i = 0u; i < n; ++i) {
      const size_t i2 = (i * 2u);
      buf[i2] = static_cast<double>(cD[i]);
      buf[i2 + 1u] = static_cast<double>(cJ[i]);
    }
    if (MPI_File_write_at(fh, (oqV + j * n * 2u), buf, (n * 2u), MPI_DOUBLE, MPI_STATUS_IGNORE)) {
      DIE("MPI_File_write_at(Z)q");
    }
  }
  if (MPI_File_sync(fh)) {
    DIE("MPI_File_sync(Z)q");
  }
#else // !USE_COMPLEX
  for (size_t j = 0u; j < n_p; ++j) {
    if (MPI_File_write_at(fh, (opV + j * n), (hV + ldhV * j), n, MPI_DOUBLE, MPI_STATUS_IGNORE)) {
      DIE("MPI_File_write_at(Z)p");
    }
  }
  if (MPI_File_sync(fh)) {
    DIE("MPI_File_sync(Z)p");
  }
  for (size_t j = 0u; j < n_q; ++j) {
    if (MPI_File_write_at(fh, (oqV + j * n), (hV + ldhV * (n_col + j)), n, MPI_DOUBLE, MPI_STATUS_IGNORE)) {
      DIE("MPI_File_write_at(Z)q");
    }
  }
  if (MPI_File_sync(fh)) {
    DIE("MPI_File_sync(Z)q");
  }
#endif // ?USE_COMPLEX
  if (MPI_File_close(&fh)) {
    DIE("MPI_File_close(Z)");
  }

  if (MPI_File_open(MPI_COMM_WORLD, strcat(strcpy(fn, ca_fn), ".SS"), (MPI_MODE_WRONLY | MPI_MODE_CREATE), MPI_INFO_NULL, &fh)) {
    DIE("MPI_File_open(SS)");
  }
  if (MPI_File_set_size(fh, (n * sizeof(double)))) {
    DIE("MPI_File_set_size(SS)");
  }
  if (MPI_File_write_at(fh, opS, hS, n_p, MPI_DOUBLE, MPI_STATUS_IGNORE)) {
    DIE("MPI_File_write_at(SS)p");
  }
  if (MPI_File_sync(fh)) {
    DIE("MPI_File_sync(SS)p");
  }
  if (MPI_File_write_at(fh, oqS, (hS + n_col), n_q, MPI_DOUBLE, MPI_STATUS_IGNORE)) {
    DIE("MPI_File_write_at(SS)q");
  }
  if (MPI_File_sync(fh)) {
    DIE("MPI_File_sync(SS)q");
  }
  if (MPI_File_close(&fh)) {
    DIE("MPI_File_close(SS)");
  }

  if (MPI_File_open(MPI_COMM_WORLD, strcat(strcpy(fn, ca_fn), ".SY"), (MPI_MODE_WRONLY | MPI_MODE_CREATE), MPI_INFO_NULL, &fh)) {
    DIE("MPI_File_open(SY)");
  }
  if (MPI_File_set_size(fh, (n * sizeof(double)))) {
    DIE("MPI_File_set_size(SY)");
  }
  if (MPI_File_write_at(fh, opS, hH, n_p, MPI_DOUBLE, MPI_STATUS_IGNORE)) {
    DIE("MPI_File_write_at(SY)p");
  }
  if (MPI_File_sync(fh)) {
    DIE("MPI_File_sync(SY)p");
  }
  if (MPI_File_write_at(fh, oqS, (hH + n_col), n_q, MPI_DOUBLE, MPI_STATUS_IGNORE)) {
    DIE("MPI_File_write_at(SY)q");
  }
  if (MPI_File_sync(fh)) {
    DIE("MPI_File_sync(SY)q");
  }
  if (MPI_File_close(&fh)) {
    DIE("MPI_File_close(SY)");
  }

  if (MPI_File_open(MPI_COMM_WORLD, strcat(strcpy(fn, ca_fn), ".SW"), (MPI_MODE_WRONLY | MPI_MODE_CREATE), MPI_INFO_NULL, &fh)) {
    DIE("MPI_File_open(SW)");
  }
  if (MPI_File_set_size(fh, (n * sizeof(double)))) {
    DIE("MPI_File_set_size(SW)");
  }
  if (MPI_File_write_at(fh, opS, hK, n_p, MPI_DOUBLE, MPI_STATUS_IGNORE)) {
    DIE("MPI_File_write_at(SW)p");
  }
  if (MPI_File_sync(fh)) {
    DIE("MPI_File_sync(SW)p");
  }
  if (MPI_File_write_at(fh, oqS, (hK + n_col), n_q, MPI_DOUBLE, MPI_STATUS_IGNORE)) {
    DIE("MPI_File_write_at(SW)q");
  }
  if (MPI_File_sync(fh)) {
    DIE("MPI_File_sync(SW)q");
  }
  if (MPI_File_close(&fh)) {
    DIE("MPI_File_close(SW)");
  }

  if (hK)
    CUDA_CALL(cudaFreeHost(hK));
  if (hH)
    CUDA_CALL(cudaFreeHost(hH));
  if (hS)
    CUDA_CALL(cudaFreeHost(hS));
#ifdef USE_COMPLEX
  if (hVJ)
    CUDA_CALL(cudaFreeHost(hVJ));
  if (hVD)
    CUDA_CALL(cudaFreeHost(hVD));
  if (hGJ)
    CUDA_CALL(cudaFreeHost(hGJ));
  if (hGD)
    CUDA_CALL(cudaFreeHost(hGD));
  if (hFJ)
    CUDA_CALL(cudaFreeHost(hFJ));
  if (hFD)
    CUDA_CALL(cudaFreeHost(hFD));
#else // !USE_COMPLEX
  if (hV)
    CUDA_CALL(cudaFreeHost(hV));
  if (hG)
    CUDA_CALL(cudaFreeHost(hG));
  if (hF)
    CUDA_CALL(cudaFreeHost(hF));
#endif // ?USE_COMPLEX

#ifdef USE_COMPLEX
  free(buf);
#endif // USE_COMPLEX
  free(fn);  
  free_strats();

  // for profiling
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaDeviceReset());

  return fini_MPI();
}
