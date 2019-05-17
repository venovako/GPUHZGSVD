#include "mpi_helper.hpp"

#include "cuda_helper.hpp"
#include "my_utils.hpp"

#ifdef OMPI_MPI_H
#include <mpi-ext.h>
#endif // OMPI_MPI_H

int init_MPI(int *const argc, char ***const argv) throw()
{
  if (!argc)
    return -1;
  if (!argv)
    return -2;
  int i = 0, f = 0, e = MPI_SUCCESS;
  if ((e = MPI_Initialized(&i)))
    return e;
  if (i)
    return MPI_SUCCESS;
  if ((e = MPI_Finalized(&f)))
    return e;
  if (f)
    return -3;
  return MPI_Init(argc, argv);
}

int fini_MPI() throw()
{
  int f = 0, e = MPI_SUCCESS;
  if ((e = MPI_Finalized(&f)))
    return e;
  if (f)
    return MPI_SUCCESS;
  return MPI_Finalize();
}

bool mpi_cuda() throw()
{
#if (defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT)
  return (1 == MPIX_Query_cuda_support());
#elif (defined(MVAPICH2_NUMVERSION) && (MVAPICH2_NUMVERSION >= 20000000))
  const char *const e = getenv("MV2_USE_CUDA");
  return (e && atoi(e));
#else // only OpenMPI and MVAPICH2 so far
  return false;
#endif // TODO: any other MPI?
}

#ifndef DEV_HOST_NAME_LEN
#define DEV_HOST_NAME_LEN 256u
#endif // !DEV_HOST_NAME_LEN

typedef struct {
  char host[DEV_HOST_NAME_LEN];
  int rank;
  int dev_count;
  int dev;
} dev_host;

static int dev_host_cmp(const dev_host *const a, const dev_host *const b) throw()
{
  assert(a);
  assert(b);
  if (a == b)
    return 0;
  const int hc = strcmp(a->host, b->host);
  if (hc < 0)
    return -1;
  if (hc > 0)
    return 1;
  if (a->rank < b->rank)
    return -2;
  if (a->rank > b->rank)
    return 2;
  if (a->dev_count < b->dev_count)
    return -3;
  if (a->dev_count > b->dev_count)
    return 3;
  if (a->dev < b->dev)
    return -4;
  if (a->dev > b->dev)
    return 4;
  return 0;
}

static int dev_host_get(dev_host &dh, const int rank) throw()
{
  (void)memset(&dh, 0, sizeof(dh));
  dh.rank = rank;
  dh.dev = -1;
  if (gethostname(dh.host, DEV_HOST_NAME_LEN - 1u))
    return (dh.dev_count = -2);
  if (cudaGetDeviceCount(&(dh.dev_count)) != cudaSuccess)
    return (dh.dev_count = -1);
  return dh.dev_count;
}

static dev_host *get_dev_hosts(int &size, int &rank) throw()
{
  size = -1;
  rank = -1;

  SYSI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));  
  SYSI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  dev_host my;
  if (dev_host_get(my, rank) <= 0)
    (void)fprintf(stderr, "Cannot query the host information (%d)\n", my.dev_count);

  dev_host *const rcv = static_cast<dev_host*>(malloc(static_cast<unsigned>(size) * sizeof(dev_host)));
  SYSP_CALL(rcv);

  SYSI_CALL(MPI_Allgather(&my, static_cast<int>(sizeof(dev_host)), MPI_BYTE, rcv, static_cast<int>(sizeof(dev_host)), MPI_BYTE, MPI_COMM_WORLD));

  if (!rank)
    (void)fprintf(stderr, "RANK,GPUS,HOSTNAME\n");
  for (int i = 0; i < size; ++i) {
    if (!rank)
      (void)fprintf(stderr, "%4d,%4d,%s\n", rcv[i].rank, rcv[i].dev_count, rcv[i].host);
    if (rcv[i].dev_count <= 0) {
      free(rcv);
      return static_cast<dev_host*>(NULL);
    }
  }

  return rcv;
}

int assign_dev2host() throw()
{
  int size = -1, rank = -1;
  dev_host *const dh = get_dev_hosts(size, rank);
  if (!dh)
    return -1;

  if (size > 1)
    qsort(dh, static_cast<size_t>(size), sizeof(dev_host), (int (*)(const void*, const void*))dev_host_cmp);

  int dev = -3;
  if (!rank)
    (void)fprintf(stderr, "\nRANK,GPUS,LGPU,HOSTNAME\n");
  dh[0].dev = 0;
  if (!rank)
    (void)fprintf(stderr, "%4d,%4d,%4d,%s\n", dh[0].rank, dh[0].dev_count, dh[0].dev, dh[0].host);
  if (dh[0].rank == rank)
    dev = dh[0].dev;

  for (int i = 1; i < size; ++i) {
    int err = 0;
    if (!strcmp(dh[i].host, dh[i-1].host)) {
      // inconsistent data
      if (dh[i].dev_count != dh[i-1].dev_count)
        err = -2;
      dh[i].dev = (dh[i-1].dev + 1);
      // more processes than devices per host, wrap around
      if (dh[i].dev >= dh[i].dev_count)
        dh[i].dev = 0;
    }
    else
      dh[i].dev = 0;
    if (!rank)
      (void)fprintf(stderr, "%4d,%4d,%4d,%s\n", dh[i].rank, dh[i].dev_count, dh[i].dev, dh[i].host);
    if (err) {
      dev = err;
      goto end;
    }
    if (dh[i].rank == rank)
      dev = dh[i].dev;
  }

 end:
  free(dh);
  return dev;
}
