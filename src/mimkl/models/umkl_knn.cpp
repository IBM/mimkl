#include <mimkl/definitions.hpp>
#include <mimkl/models/umkl_knn.hpp>

namespace mimkl
{
namespace models
{

template class UMKLKNN<double, mimkl::definitions::Kernel>;

} // namespace models
} // namespace mimkl
