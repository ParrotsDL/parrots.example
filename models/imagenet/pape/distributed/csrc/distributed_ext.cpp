#include <math.h>
#include <assert.h>
#include <iostream>
#include <algorithm>
#include <nccl.h>
#include <mpi.h>
#include <stdio.h>

#include <ATen/ATen.h>
#include <THC/THC.h>

#if (PYTORCH_VERSION >= 10000)
    #include <torch/extension.h>
    #define GET_CURRENT_STREAM at::cuda::getCurrentCUDAStream()
#else
    #include <torch/torch.h>
    #define GET_CURRENT_STREAM at::CUDAStream(at::detail::CUDAStream_getAndRetainCurrentStream())
#endif

#if (PYTORCH_VERSION >= 10100)
    #define GET_FREE_MUTEX c10::cuda::CUDACachingAllocator::getFreeMutex
#else
    #define GET_FREE_MUTEX THCCachingAllocator_getCudaFreeMutex
#endif

using namespace at;
using namespace std;

inline void hash_combine(size_t& seed) { }

template <typename T, typename... Rest>
inline void hash_combine(size_t& seed, const T& v, Rest... rest) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  hash_combine(seed, rest...);
}

#define MAKE_HASHABLE(type, ...)                      \
  namespace std {                                     \
    template<> struct hash<type> {                    \
      size_t operator()(const type &t) const {        \
        size_t ret = 0;                               \
        hash_combine(ret, __VA_ARGS__);               \
        return ret;                                   \
      }                                               \
    };                                                \
  }

MAKE_HASHABLE(at::ScalarType, static_cast<int>(t));

namespace pape {

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_CPU_INPUT(x) CHECK_CONTIGUOUS(x)

#define MPICHECK(cmd) do{                             \
    int r = cmd;                                      \
    if (r!= MPI_SUCCESS) {                            \
        fprintf(stderr, "MPI failure %s:%d '%d'\n",   \
            __FILE__,__LINE__,r);                     \
        exit(EXIT_FAILURE);                           \
    }                                                 \
}while(0)

#define NCCLCHECK(cmd) do {                           \
    ncclResult_t r = cmd;                             \
    if (r!= ncclSuccess) {                            \
        fprintf(stderr, "NCCL failure %s:%d '%s'\n",  \
            __FILE__,__LINE__,ncclGetErrorString(r)); \
        exit(EXIT_FAILURE);                           \
    }                                                 \
} while(0)

#define CUDACHECK(cmd) do {                           \
    cudaError_t r = cmd;                              \
    if (r!= cudaSuccess) {                            \
        fprintf(stderr, "CUDA failure %s:%d '%s'\n",  \
            __FILE__,__LINE__,cudaGetErrorString(r)); \
        exit(EXIT_FAILURE);                           \
    }                                                 \
} while (0)

struct CommGroup {
    int rank;
    int local_rank;
    int world_size;

    MPI_Comm mpi_comm;
    ncclComm_t nccl_comm;
};

struct Backend {
    static const int AUTO = 0;
    static const int MPI = 1;
    static const int NCCL = 2;
};

std::unordered_map<int, ncclRedOp_t> ncclOp = {
  {0, ncclSum},
  {1, ncclProd},
  {2, ncclMin},
  {3, ncclMax},
};

std::unordered_map<at::ScalarType, ncclDataType_t> ncclDatatype = {
  {at::kChar, ncclInt8},
  {at::kByte, ncclUint8},
  {at::kFloat, ncclFloat},
  {at::kDouble, ncclDouble},
  {at::kInt, ncclInt32},
  {at::kLong, ncclInt64},
  {at::kHalf, ncclHalf},
};

std::unordered_map<int, MPI_Op> mpiOp = {
  {0, MPI_SUM},
  {1, MPI_PROD},
  {2, MPI_MIN},
  {3, MPI_MAX},
};

std::unordered_map<at::ScalarType, MPI_Datatype> mpiDatatype = {
  {at::kByte, MPI_UNSIGNED_CHAR},
  {at::kChar, MPI_CHAR},
  {at::kDouble, MPI_DOUBLE},
  {at::kFloat, MPI_FLOAT},
  {at::kInt, MPI_INT},
  {at::kLong, MPI_LONG},
  {at::kShort, MPI_SHORT},
};

std::unordered_map<int, CommGroup> groups_;

int nccl_init(CommGroup& wg) {

    ncclUniqueId commId;

    // NCCL Communicator creation
    CUDACHECK(cudaDeviceSynchronize());
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    NCCLCHECK(ncclGetUniqueId(&commId));
    MPICHECK(MPI_Bcast(&commId, sizeof(commId), MPI_BYTE, 0, MPI_COMM_WORLD));
    NCCLCHECK(ncclCommInitRank(&wg.nccl_comm, wg.world_size, commId, wg.rank));

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    return 0;
}

int mpi_init(CommGroup& wg) {
    int provided;
    MPICHECK(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided));
    if(provided != MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "MPI init failure: not support multiple thread\n");
        exit(EXIT_FAILURE);                           \
    }
    wg.mpi_comm = MPI_COMM_WORLD;
    MPI_Comm_rank(MPI_COMM_WORLD, &wg.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &wg.world_size);

    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &shmcomm);
    MPI_Comm_rank(shmcomm, &wg.local_rank);

    return 0; 
}

void initialize() {
    CommGroup world_group;
    mpi_init(world_group);
    CUDACHECK(cudaSetDevice(world_group.local_rank));
    nccl_init(world_group);
    
    groups_.insert({0, world_group});
}

void barrier(int gid) {
    AT_ASSERTM(groups_.find(gid) != groups_.end(), "group not exist!");
    if (groups_[gid].mpi_comm != MPI_COMM_NULL) {
        MPICHECK(MPI_Barrier(groups_[gid].mpi_comm));
    }
}

void finalize() {
    for (auto &g : groups_) {
        if (g.second.mpi_comm != MPI_COMM_NULL) {
            CUDACHECK(cudaDeviceSynchronize());
            NCCLCHECK(ncclCommDestroy(g.second.nccl_comm));
            if (g.second.mpi_comm != MPI_COMM_WORLD) {
                MPI_Comm_free(&g.second.mpi_comm);
            }
        }
    }
    // MPI maybe stuck here if one process failed but others not
    MPI_Finalize();
}

int get_rank(int gid) {
    AT_ASSERTM(groups_.find(gid) != groups_.end(), "group not exist!");
    return groups_[gid].rank;
}

int get_local_rank(void) {
    AT_ASSERTM(groups_.find(0) != groups_.end(), "group not exist!");
    return groups_[0].local_rank;
}

int get_world_size(int gid) {
    AT_ASSERTM(groups_.find(gid) != groups_.end(), "group not exist!");
    return groups_[gid].world_size;
}

int new_group(const std::vector<int>& ranks) {

    int color = MPI_UNDEFINED;
    int cur_rank = groups_[0].rank;
    if (ranks.size() > 0) {
        color = *min_element(ranks.begin(), ranks.end());
        if (std::find(ranks.begin(), ranks.end(), cur_rank) == ranks.end()) {
            std::string errMsg = "New group parameter should be [] or include itself. ";
            errMsg += "Rank: " + std::to_string(cur_rank) + " group ranks: [";
            for (auto r : ranks) {
                errMsg += std::to_string(r) + ",";
            }
            errMsg += "]";
            AT_ERROR(errMsg);
        }
    }

    MPI_Comm new_comm_mpi;
    MPICHECK(MPI_Comm_split(MPI_COMM_WORLD, color, cur_rank, &new_comm_mpi));

    CommGroup new_group;
    new_group.mpi_comm = new_comm_mpi;

    if (new_group.mpi_comm != MPI_COMM_NULL) {
        MPICHECK(MPI_Comm_rank(new_group.mpi_comm, &new_group.rank));
        MPICHECK(MPI_Comm_size(new_group.mpi_comm, &new_group.world_size));
        new_group.local_rank = -1;
        AT_ASSERTM((int)ranks.size() == new_group.world_size, "New group size is not"
                   "equal with ranks parameter" + std::to_string(new_group.world_size) + 
                   "vs" + std::to_string(ranks.size()) + "@" + std::to_string(cur_rank));

        CUDACHECK(cudaDeviceSynchronize());
        MPICHECK(MPI_Barrier(new_group.mpi_comm));
        ncclUniqueId ncclId;
        NCCLCHECK(ncclGetUniqueId(&ncclId));
        MPICHECK(MPI_Bcast((void *)&ncclId, sizeof(ncclId), MPI_BYTE, 0, new_group.mpi_comm)); 
        NCCLCHECK(ncclCommInitRank(&new_group.nccl_comm, new_group.world_size, ncclId, new_group.rank));;
        MPICHECK(MPI_Barrier(new_group.mpi_comm));
    }

    int new_group_id = groups_.size();
    groups_.insert({new_group_id, new_group});
  
    return new_group_id;
}

void all_reduce(Tensor tensor, int op, int group, int backend) {
    const auto& g = groups_.at(group);
    if (g.mpi_comm == MPI_COMM_NULL)
        return;

    if (tensor.type().is_cuda()) {
        CHECK_CUDA_INPUT(tensor);
        if (backend == Backend::AUTO || backend == Backend::NCCL) {
            std::unique_lock<std::mutex> cudaFreeMutexLock(
                *(GET_FREE_MUTEX()));

            NCCLCHECK(ncclAllReduce(tensor.data_ptr(), tensor.data_ptr(), tensor.numel(),
                ncclDatatype.at(tensor.type().scalarType()), ncclOp.at(op), g.nccl_comm,
                GET_CURRENT_STREAM));
            cudaFreeMutexLock.unlock();
        } else {
            MPICHECK(MPI_Allreduce(MPI_IN_PLACE, tensor.data_ptr(), tensor.numel(),
                mpiDatatype.at(tensor.type().scalarType()), mpiOp.at(op), g.mpi_comm));
        }
    } else {
        CHECK_CPU_INPUT(tensor);
        if (backend == Backend::AUTO || backend == Backend::MPI) {
            MPICHECK(MPI_Allreduce(MPI_IN_PLACE, tensor.data_ptr(), tensor.numel(),
                mpiDatatype.at(tensor.type().scalarType()), mpiOp.at(op), g.mpi_comm));
        } else {
            AT_ERROR("CPU data cannot use NCCL backend!");
        }
    }
}

void broadcast(Tensor tensor, int src, int group, int backend) {
    const auto& g = groups_.at(group);
    if (g.mpi_comm == MPI_COMM_NULL)
        return;

    if (tensor.type().is_cuda()) {
        CHECK_CUDA_INPUT(tensor);
        if (backend == Backend::AUTO || backend == Backend::NCCL) {
            std::unique_lock<std::mutex> cudaFreeMutexLock(
                *(GET_FREE_MUTEX()));

            NCCLCHECK(ncclBcast(tensor.data_ptr(), tensor.numel(),
                ncclDatatype.at(tensor.type().scalarType()), src,
                g.nccl_comm, GET_CURRENT_STREAM));
            cudaFreeMutexLock.unlock();
        } else {
            MPICHECK(MPI_Bcast(tensor.data_ptr(), tensor.numel(),
                mpiDatatype.at(tensor.type().scalarType()), src, g.mpi_comm));
        }
    } else {
        CHECK_CPU_INPUT(tensor);
        if (backend == Backend::AUTO || backend == Backend::MPI) {
            MPICHECK(MPI_Bcast(tensor.data_ptr(), tensor.numel(),
                mpiDatatype.at(tensor.type().scalarType()), src, g.mpi_comm));
        } else {
            AT_ERROR("CPU data cannot use NCCL backend!");
        }
    }
}

void reduce(Tensor tensor, int dst, int op, int group, int backend) {
    const auto& g = groups_.at(group);
    if (g.mpi_comm == MPI_COMM_NULL)
        return;

    if (tensor.type().is_cuda()) {
        CHECK_CUDA_INPUT(tensor);
        if (backend == Backend::AUTO || backend == Backend::NCCL) {
            std::unique_lock<std::mutex> cudaFreeMutexLock(
                *(GET_FREE_MUTEX()));

            NCCLCHECK(ncclReduce(tensor.data_ptr(), tensor.data_ptr(), tensor.numel(),
                ncclDatatype.at(tensor.type().scalarType()), ncclOp.at(op), dst,
                g.nccl_comm, GET_CURRENT_STREAM));
            cudaFreeMutexLock.unlock();
        } else {
            if (g.rank == dst) {
                MPICHECK(MPI_Reduce(MPI_IN_PLACE, tensor.data_ptr(), tensor.numel(),
                    mpiDatatype.at(tensor.type().scalarType()), mpiOp.at(op), dst, g.mpi_comm));
            } else {
                MPICHECK(MPI_Reduce(tensor.data_ptr(), NULL, tensor.numel(),
                    mpiDatatype.at(tensor.type().scalarType()), mpiOp.at(op), dst, g.mpi_comm));
            }
        }
    } else {
        CHECK_CPU_INPUT(tensor);
        if (backend == Backend::AUTO || backend == Backend::MPI) {
            if (g.rank == dst) {
                MPICHECK(MPI_Reduce(MPI_IN_PLACE, tensor.data_ptr(), tensor.numel(),
                    mpiDatatype.at(tensor.type().scalarType()), mpiOp.at(op), dst, g.mpi_comm));
            } else {
                MPICHECK(MPI_Reduce(tensor.data_ptr(), NULL, tensor.numel(),
                    mpiDatatype.at(tensor.type().scalarType()), mpiOp.at(op), dst, g.mpi_comm));
            }
        } else {
            AT_ERROR("CPU data cannot use NCCL backend!");
        }
    }
}

void all_gather(Tensor tensor_send, Tensor tensor_recv, int group, int backend) {
    const auto& g = groups_.at(group);
    if (g.mpi_comm == MPI_COMM_NULL)
        return;

    if (tensor_send.type().is_cuda()) {
        CHECK_CUDA_INPUT(tensor_send);
        CHECK_CUDA_INPUT(tensor_recv);
        if (backend == Backend::AUTO || backend == Backend::NCCL) {
            std::unique_lock<std::mutex> cudaFreeMutexLock(
                *(GET_FREE_MUTEX()));

            NCCLCHECK(ncclAllGather(tensor_send.data_ptr(), tensor_recv.data_ptr(),
                tensor_send.numel(), ncclDatatype.at(tensor_send.type().scalarType()),
                g.nccl_comm, GET_CURRENT_STREAM));
            cudaFreeMutexLock.unlock();
        } else {
            MPICHECK(MPI_Allgather(tensor_send.data_ptr(), tensor_send.numel(),
                mpiDatatype.at(tensor_send.type().scalarType()), tensor_recv.data_ptr(),
                tensor_send.numel(), mpiDatatype.at(tensor_recv.type().scalarType()),
                g.mpi_comm));
        }
    } else {
        CHECK_CPU_INPUT(tensor_send);
        CHECK_CPU_INPUT(tensor_recv);
        if (backend == Backend::AUTO || backend == Backend::MPI) {
            MPICHECK(MPI_Allgather(tensor_send.data_ptr(), tensor_send.numel(),
                mpiDatatype.at(tensor_send.type().scalarType()), tensor_recv.data_ptr(),
                tensor_send.numel(), mpiDatatype.at(tensor_recv.type().scalarType()),
                g.mpi_comm));
        } else {
            AT_ERROR("CPU data cannot use NCCL backend!");
        }
    }
}

void reduce_scatter(Tensor tensor_send, Tensor tensor_recv, int op, int group, int backend) {
    const auto& g = groups_.at(group);
    if (g.mpi_comm == MPI_COMM_NULL)
        return;

    if (tensor_send.type().is_cuda()) {
        CHECK_CUDA_INPUT(tensor_send);
        CHECK_CUDA_INPUT(tensor_recv);
        if (backend == Backend::AUTO || backend == Backend::NCCL) {
            std::unique_lock<std::mutex> cudaFreeMutexLock(
                *(GET_FREE_MUTEX()));

            NCCLCHECK(ncclReduceScatter(tensor_send.data_ptr(), tensor_recv.data_ptr(), tensor_recv.numel(),
                ncclDatatype.at(tensor_send.type().scalarType()), ncclOp.at(op), g.nccl_comm,
                GET_CURRENT_STREAM));
            cudaFreeMutexLock.unlock();
        } else {
            MPICHECK(MPI_Reduce_scatter_block(tensor_send.data_ptr(), tensor_recv.data_ptr(), tensor_recv.numel(),
                mpiDatatype.at(tensor_send.type().scalarType()), mpiOp.at(op), g.mpi_comm));
        }
    } else {
        CHECK_CPU_INPUT(tensor_send);
        CHECK_CPU_INPUT(tensor_recv);
        if (backend == Backend::AUTO || backend == Backend::MPI) {
            MPICHECK(MPI_Reduce_scatter_block(tensor_send.data_ptr(), tensor_recv.data_ptr(), tensor_recv.numel(),
                mpiDatatype.at(tensor_send.type().scalarType()), mpiOp.at(op), g.mpi_comm));
        } else {
            AT_ERROR("CPU data cannot use NCCL backend!");
        }
    }
}


void gather(Tensor tensor_send, Tensor tensor_recv, int dst, int group, int backend) {
    const auto& g = groups_.at(group);
    if (g.mpi_comm == MPI_COMM_NULL)
        return;

    if (tensor_send.type().is_cuda()) {
        CHECK_CUDA_INPUT(tensor_send);
        CHECK_CUDA_INPUT(tensor_recv);
        if (backend == Backend::AUTO || backend == Backend::MPI) {
            MPICHECK(MPI_Gather(tensor_send.data_ptr(), tensor_send.numel(), mpiDatatype.at(tensor_send.type().scalarType()),
                tensor_recv.data_ptr(), tensor_send.numel(), mpiDatatype.at(tensor_recv.type().scalarType()), dst, 
                g.mpi_comm));
        } else {
            AT_ERROR("Cannot use NCCL backend!");
        }
    } else {
        CHECK_CPU_INPUT(tensor_send);
        CHECK_CPU_INPUT(tensor_recv);
        if (backend == Backend::AUTO || backend == Backend::MPI) {
            MPICHECK(MPI_Gather(tensor_send.data_ptr(), tensor_send.numel(), mpiDatatype.at(tensor_send.type().scalarType()),
                tensor_recv.data_ptr(), tensor_send.numel(), mpiDatatype.at(tensor_recv.type().scalarType()), dst,
                g.mpi_comm));
        } else {
            AT_ERROR("Cannot use NCCL backend!");
        }
    }
}


void scatter(Tensor tensor_send, Tensor tensor_recv, int src, int group, int backend) {
    const auto& g = groups_.at(group);
    if (g.mpi_comm == MPI_COMM_NULL)
        return;

    if (tensor_send.type().is_cuda()) {
        CHECK_CUDA_INPUT(tensor_send);
        CHECK_CUDA_INPUT(tensor_recv);
        if (backend == Backend::AUTO || backend == Backend::MPI) {
            MPICHECK(MPI_Scatter(tensor_send.data_ptr(), tensor_recv.numel(), mpiDatatype.at(tensor_send.type().scalarType()),
                tensor_recv.data_ptr(), tensor_recv.numel(), mpiDatatype.at(tensor_recv.type().scalarType()), src, 
                g.mpi_comm));
        } else {
            AT_ERROR("Cannot use NCCL backend!");
        }
    } else {
        CHECK_CPU_INPUT(tensor_send);
        CHECK_CPU_INPUT(tensor_recv);
        if (backend == Backend::AUTO || backend == Backend::MPI) {
            MPICHECK(MPI_Scatter(tensor_send.data_ptr(), tensor_recv.numel(), mpiDatatype.at(tensor_send.type().scalarType()),
                tensor_recv.data_ptr(), tensor_recv.numel(), mpiDatatype.at(tensor_recv.type().scalarType()), src,
                g.mpi_comm));
        } else {
            AT_ERROR("Cannot use NCCL backend!");
        }
    }
}

void send(Tensor tensor, int dst, int group, int tag, int backend) {
    const auto& g = groups_.at(group);
    if (g.mpi_comm == MPI_COMM_NULL)
        return;

    if (tensor.type().is_cuda()) {
        CHECK_CUDA_INPUT(tensor);
        if (backend == Backend::AUTO || backend == Backend::MPI) {
            MPICHECK(MPI_Send(tensor.data_ptr(), tensor.numel(), mpiDatatype.at(tensor.type().scalarType()),
                dst, tag, g.mpi_comm));
        } else {
            AT_ERROR("Cannot use NCCL backend!");
        }
    } else {
        CHECK_CPU_INPUT(tensor);
        if (backend == Backend::AUTO || backend == Backend::MPI) {
            MPICHECK(MPI_Send(tensor.data_ptr(), tensor.numel(), mpiDatatype.at(tensor.type().scalarType()),
                dst, tag, g.mpi_comm));
        } else {
            AT_ERROR("Cannot use NCCL backend!");
        }
    }
}

void recv(Tensor tensor, int src, int group, int tag, int backend) {
    const auto& g = groups_.at(group);
    if (g.mpi_comm == MPI_COMM_NULL)
        return;

    if (tensor.type().is_cuda()) {
        CHECK_CUDA_INPUT(tensor);
        if (backend == Backend::AUTO || backend == Backend::MPI) {
            MPICHECK(MPI_Recv(tensor.data_ptr(), tensor.numel(), mpiDatatype.at(tensor.type().scalarType()),
                src, tag, g.mpi_comm, MPI_STATUS_IGNORE));
        } else {
            AT_ERROR("Cannot use NCCL backend!");
        }
    } else {
        CHECK_CPU_INPUT(tensor);
        if (backend == Backend::AUTO || backend == Backend::MPI) {
            MPICHECK(MPI_Recv(tensor.data_ptr(), tensor.numel(), mpiDatatype.at(tensor.type().scalarType()),
                src, tag, g.mpi_comm, MPI_STATUS_IGNORE));
        } else {
            AT_ERROR("Cannot use NCCL backend!");
        }
    }
}

int recv_all(Tensor tensor, int group, int tag, int backend) {
    const auto& g = groups_.at(group);
    if (g.mpi_comm == MPI_COMM_NULL)
        return -1;

    MPI_Status status;
    if (tensor.type().is_cuda()) {
        CHECK_CUDA_INPUT(tensor);
        if (backend == Backend::AUTO || backend == Backend::MPI) {
            MPICHECK(MPI_Recv(tensor.data_ptr(), tensor.numel(), mpiDatatype.at(tensor.type().scalarType()),
                MPI_ANY_SOURCE, tag, g.mpi_comm, &status));
        } else {
            AT_ERROR("Cannot use NCCL backend!");
        }
    } else {
        CHECK_CPU_INPUT(tensor);
        if (backend == Backend::AUTO || backend == Backend::MPI) {
            MPICHECK(MPI_Recv(tensor.data_ptr(), tensor.numel(), mpiDatatype.at(tensor.type().scalarType()),
                MPI_ANY_SOURCE, tag, g.mpi_comm, &status));
        } else {
            AT_ERROR("Cannot use NCCL backend!");
        }
    }
    return status.MPI_SOURCE;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("initialize", &initialize);
  m.def("finalize", &finalize);
  m.def("barrier", &barrier,
        py::arg("group"));
  m.def("new_group", &new_group,
        py::arg("ranks"));
  m.def("get_rank", &get_rank,
        py::arg("group"));
  m.def("get_world_size", &get_world_size,
        py::arg("group"));
  m.def("get_local_rank", &get_local_rank);
  m.def("all_reduce", &all_reduce,
        py::arg("tensor"),
        py::arg("op"),
        py::arg("group"),
        py::arg("backend"));
  m.def("broadcast", &broadcast,
        py::arg("tensor"),
        py::arg("src"),
        py::arg("group"),
        py::arg("backend"));
  m.def("reduce", &reduce,
        py::arg("tensor"),
        py::arg("dst"),
        py::arg("op"),
        py::arg("group"),
        py::arg("backend"));
  m.def("all_gather", &all_gather,
        py::arg("tensor_send"),
        py::arg("tensor_recv"),
        py::arg("group"),
        py::arg("backend"));
  m.def("reduce_scatter", &reduce_scatter,
        py::arg("tensor_send"),
        py::arg("tensor_recv"),
        py::arg("op"),
        py::arg("group"),
        py::arg("backend"));
  m.def("gather", &gather,
        py::arg("tensor_send"),
        py::arg("tensor_recv"),
        py::arg("dst"),
        py::arg("group"),
        py::arg("backend"));
  m.def("scatter", &scatter,
        py::arg("tensor_send"),
        py::arg("tensor_recv"),
        py::arg("src"),
        py::arg("group"),
        py::arg("backend"));
  m.def("send", &send,
        py::arg("tensor"),
        py::arg("dst"),
        py::arg("group"),
        py::arg("tag"),
        py::arg("backend"));
  m.def("recv", &recv,
        py::arg("tensor"),
        py::arg("src"),
        py::arg("group"),
        py::arg("tag"),
        py::arg("backend"));
  m.def("recv_all", &recv_all,
        py::arg("tensor"),
        py::arg("group"),
        py::arg("tag"),
        py::arg("backend"));
}

} // namespace pape
