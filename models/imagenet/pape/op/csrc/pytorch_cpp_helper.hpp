#ifndef PYTORCH_CPP_HELPER
#define PYTORCH_CPP_HELPER

#if (PYTORCH_VERSION < 1000)
    #include <torch/torch.h>
#else
    #include <torch/extension.h>
#endif

#endif // PYTORCH_CPP_HELPER
