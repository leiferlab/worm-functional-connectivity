// The only function in here is a template (needs to be usable both for doubles
// and std::complex. Its implementation has therefore to be visible in the 
// header: don't define the interface here but have the implementation in the
// .tcc file.

namespace integration {
}

#include "integration.tcc"
