#include <mimkl/data_structures/utilities.hpp>
#include <numeric>

namespace mimkl
{
namespace data_structures
{

std::ostream &operator<<(std::ostream &output_stream, const Indexing &indexing)
{
    for (auto indexing_iterator = indexing.begin();
         indexing_iterator != indexing.end();)
    {
        const std::string &key = indexing_iterator->first;
        output_stream << key << " -> (";
        output_stream << indexing_iterator->second; // first key-value pair
        ++indexing_iterator;                        // increment for-loop and
        while (indexing_iterator != indexing.end() &&
               key == indexing_iterator->first)
        { // works only if equal keys are neighbors
            output_stream << ", " << indexing_iterator->second;
            ++indexing_iterator;
        }
        output_stream << ")" << std::endl;
    }
    return output_stream;
}

std::vector<int> range(const int n)
{
    std::vector<int> a_range(n);
    std::iota(a_range.begin(), a_range.end(), 0);
    return a_range;
}

Indexing indexing_from_vector(const std::vector<int> &a_vector)
{
    Indexing an_indexing;
    for (const auto &element : a_vector)
    {
        an_indexing.emplace(std::to_string(element), element);
    }
    return an_indexing;
}
Indexing indexing_from_vector_of_strings(const std::vector<std::string> &a_vector)
{
    Indexing an_indexing;
    for (mimkl::definitions::Index i = 0; i < a_vector.size(); ++i)
    {
        an_indexing.emplace(a_vector[i], i);
    }
    return an_indexing;
}

} // namespace data_structures
} // namespace mimkl
