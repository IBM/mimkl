#include <mimkl/data_structures/data_frame.hpp>
#include <mimkl/data_structures/utilities.hpp>

namespace mimkl
{
namespace data_structures
{

void DataFrame::build_indexing_selection(const std::vector<std::string> &indices,
                                         const Indexing &original_indexing,
                                         u_int &number_of_selected_elements,
                                         std::vector<int> &selected_indices_vector,
                                         Indexing &selected_indices)
{

    // clear objects passed by reference
    number_of_selected_elements = 0;
    std::vector<int>().swap(selected_indices_vector);
    Indexing().swap(selected_indices);

    // fill the selected indexing.
    // count number of indices
    for (const auto &index : indices)
    {
        number_of_selected_elements += original_indexing.count(index);
    }
    if (number_of_selected_elements == 0)
    {
        throw "No such index found";
    }
    selected_indices_vector.reserve(number_of_selected_elements);

    Index re_index = 0; // do not carry the index of the original_index over,
                        // but assign new in order of query
    for (const auto &index : indices)
    {
        const auto index_range = original_indexing.equal_range(index);
        for (auto index_range_iterator = index_range.first;
             index_range_iterator != index_range.second; ++index_range_iterator)
        {
            selected_indices_vector.push_back(index_range_iterator->second);
            selected_indices.emplace(index_range_iterator->first,
                                     re_index++); // not *index_range_iterator
        }
    }
}

//! get the row/s corresponding to a list of indices
DataFrame DataFrame::access_by_index(const std::vector<std::string> &indices)
{
    const u_int number_of_columns = columns_.size();
    // create indices for columns
    std::vector<int> column_indices_vector = range(number_of_columns);

    // create indices for rows
    Indexing selected_indices;
    u_int number_of_selected_rows = 0;
    std::vector<int> row_indices_vector;
    build_indexing_selection(indices, index_, number_of_selected_rows,
                             row_indices_vector, selected_indices);

    // get the Eigen::ArrayXi for the indices
    Eigen::Map<Eigen::ArrayXi> row_indices(row_indices_vector.data(),
                                           number_of_selected_rows);
    Eigen::Map<Eigen::ArrayXi> column_indices(column_indices_vector.data(),
                                              number_of_columns);

    return DataFrame(mimkl::linear_algebra::indexing(this->matrix(), row_indices,
                                                     column_indices),
                     selected_indices, columns_);
}

//! Get the column/s corresponding to a list of columns
DataFrame DataFrame::access_by_columns(const std::vector<std::string> &columns)
{
    const u_int number_of_rows = index_.size();
    // create indices for rows
    std::vector<int> row_indices_vector = range(number_of_rows);

    // create indices for columns
    Indexing selected_columns;
    u_int number_of_selected_columns = 0;
    std::vector<int> column_indices_vector;
    build_indexing_selection(columns, columns_, number_of_selected_columns,
                             column_indices_vector, selected_columns);

    // get the Eigen::ArrayXi for the indices
    Eigen::Map<Eigen::ArrayXi> row_indices(row_indices_vector.data(),
                                           number_of_rows);
    Eigen::Map<Eigen::ArrayXi> column_indices(column_indices_vector.data(),
                                              number_of_selected_columns);

    return DataFrame(mimkl::linear_algebra::indexing(this->matrix(), row_indices,
                                                     column_indices),
                     index_, selected_columns);
}

//! dump a DataFrame to std::ostream
std::ostream &operator<<(std::ostream &output_stream, const DataFrame &data_frame)
{
    output_stream << "Index:" << std::endl;
    output_stream << data_frame.index_;
    output_stream << "Columns:" << std::endl;
    output_stream << data_frame.columns_;
    output_stream << "Values:" << std::endl;
    output_stream << data_frame.matrix() << std::endl;
    return output_stream;
}

//! get the row/s corresponding to the given index
DataFrame DataFrame::loc(const std::string &index)
{
    return this->access_by_index({index});
}

//! get the row/s corresponding to a list of indices
DataFrame DataFrame::loc(const std::vector<std::string> &indices)
{
    return this->access_by_index(indices);
}

//! get the column/s corresponding to the given column
DataFrame DataFrame::operator[](const std::string &column)
{
    return this->access_by_columns({column});
}

//! get the column/s corresponding to a list of columns
DataFrame DataFrame::operator[](const std::vector<std::string> &columns)
{
    return this->access_by_columns(columns);
}

} // namespace data_structures
} // namespace mimkl
