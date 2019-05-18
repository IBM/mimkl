#ifndef INCLUDE_MIMKL_DATA_STRUCTURES_UTILITIES_HPP_
#define INCLUDE_MIMKL_DATA_STRUCTURES_UTILITIES_HPP_

#include <mimkl/definitions.hpp>
#include <ostream>
#include <vector>

using mimkl::definitions::Index;
using mimkl::definitions::Indexing;

namespace mimkl
{
namespace data_structures
{

auto select_first = [](const auto &pair) { return pair.first; };
auto select_second = [](const auto &pair) { return pair.second; };

std::ostream &operator<<(std::ostream &os, const Indexing &);
std::vector<int> range(const int);
Indexing indexing_from_vector(const std::vector<int> &);
Indexing indexing_from_vector_of_strings(const std::vector<std::string> &);

template <typename Scalar>
MATRIX(Scalar)
dichotomies(std::vector<std::string> labels)
{
    Indexing class_map = indexing_from_vector_of_strings(labels);
    std::sort(labels.begin(), labels.end()); // pass labels as value
                                             //	  _unique_labels = labels;
    auto last = std::unique(labels.begin(), labels.end());
    labels.erase(last, labels.end());
    MATRIX(Scalar)
    D = MATRIX(Scalar)::Constant(class_map.size(), labels.size(), -1);
    for (Index j = 0; j < labels.size(); ++j)
    {
        auto range = class_map.equal_range(labels[j]);
        for (auto i = range.first; i != range.second; ++i)
        {
            D(i->second, j) = 1;
        }
    }
    return D;
}

} // namespace data_structures
} // namespace mimkl

#endif /* INCLUDE_MIMKL_DATA_STRUCTURES_UTILITIES_HPP_ */
