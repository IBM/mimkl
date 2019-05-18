#ifndef _IO_HPP_
#define _IO_HPP_

#include "mimkl/data_structures.hpp"
#include "mimkl/definitions.hpp"
#include <Eigen/Core>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using mimkl::data_structures::DataFrame;
using mimkl::definitions::Indexing;

namespace mimkl
{
namespace io
{

enum HeaderPolicy
{
    none,
    index_only,
    column_only,
    both
};

/*
 * Character Separated Values Reader
 * */
class CSVReader
{

    public:
    // costructor const char & sep=',',const char file_header_char='n'
    template <typename MatrixType>
    static mimkl::data_structures::DataFrame
    read(const std::string &path,
         const char &sep = ',',
         const HeaderPolicy header_policy = none)
    {
        typedef typename MatrixType::Scalar Value;

        // allocate variables
        Indexing index_;
        Indexing columns_;
        std::ifstream input_stream;
        input_stream.open(path);
        if (!input_stream.good())
        {
            std::runtime_error("ifstream for " + path +
                               " is not in  a valid state.");
        }
        std::string line;
        std::vector<Value> parsed_values; // requires templatization
        Index rows = 0;

        // define functions to avoid conditional checks in loops
        /*
         *                  sv_parser
         *                  /      \
         * parse_first_line  	   parse_lines (while)
         * 							 \
         * 							 line_parser
         *                          /        \
         *           parse_first_cell         parse_cells (while)
         */
        std::function<void(std::stringstream &, std::string &)> parse_first_cell;
        std::function<void(std::stringstream &, std::string &)> parse_cells;
        std::function<void(std::stringstream &, std::string &)> line_parser;
        std::function<void()> parse_first_line;
        std::function<void()> parse_lines;
        std::function<void()> sv_parser;

        parse_first_cell = [&](std::stringstream &line_stream, std::string &cell) {
            if (std::getline(line_stream, cell, sep))
            { // non empty line
                index_.emplace(cell, rows);
            };
        };

        parse_cells = [&](std::stringstream &line_stream, std::string &cell) {
            while (std::getline(line_stream, cell, sep))
            {
                parsed_values.push_back(std::stod(cell));
            }
            ++rows;
        };

        line_parser = [&](std::stringstream &line_stream,
                          std::string &cell) { // , & index_
            parse_cells(line_stream, cell);
        };
        if (header_policy == index_only || header_policy == both)
        {
            line_parser = [&](std::stringstream &line_stream, std::string &cell) {
                parse_first_cell(line_stream, cell);
                parse_cells(line_stream, cell);
            };
        };

        parse_first_line = [&]() {
            if (std::getline(input_stream, line))
            { // non empty file
                // first line
                std::vector<std::string> parsed_column_labels;
                std::stringstream line_stream(line);
                Index i = 0;
                std::string cell;
                if (header_policy == both)
                { // first cell of first line is empty if
                    // there is an index header, consume it
                    // here
                    std::getline(line_stream, cell, sep);
                };
                while (std::getline(line_stream, cell, sep))
                {
                    columns_.emplace(cell, i++);
                };
            };
        };

        parse_lines = [&]() {
            while (std::getline(input_stream, line))
            {
                std::stringstream line_stream(line);
                std::string cell;
                line_parser(line_stream, cell);
            };
        };

        //! To parse the line, apply the right reader function depending on file
        //! (file_header_char)
        sv_parser = [&]() { parse_lines(); };
        if (header_policy == column_only || header_policy == both)
        { // parse first line to Indexing, then go on
            sv_parser =
            [&]() { //  parse_first_line, &parse_lines, & line_parser, &
                //  sep, & header_policy, &rows, & parsed_values,&
                //  input_stream, & line, &index_,&columns_
                parse_first_line();
                parse_lines();
            };
        };

        // execute
        sv_parser(); // header_policy, input_stream, line, sep, rows,
                     // parsed_values,
                     // index_, columns_

        Eigen::MatrixXd matrix;
        matrix =
        Eigen::Map<const Eigen::Matrix<Value, MatrixType::RowsAtCompileTime,
                                       MatrixType::ColsAtCompileTime, Eigen::RowMajor>>(
        parsed_values.data(), rows, parsed_values.size() / rows);

        // for missing headers
        switch (header_policy)
        {
        case both:
            break;
        case index_only:
            columns_ = mimkl::data_structures::indexing_from_vector(
            mimkl::data_structures::range(matrix.cols()));
            break;
        case column_only:
            index_ = mimkl::data_structures::indexing_from_vector(
            mimkl::data_structures::range(matrix.rows()));
            break;
        case none:
            columns_ = mimkl::data_structures::indexing_from_vector(
            mimkl::data_structures::range(matrix.cols()));
            index_ = mimkl::data_structures::indexing_from_vector(
            mimkl::data_structures::range(matrix.rows()));
            break;
        default:
            throw("given header_policy is undefined");
        }

        DataFrame df(matrix, index_, columns_);
        return df;
    }
};

template <typename MatrixType>
MatrixType eigen_matrix_from_csv(const std::string &path, const char &sep = ',')
{
    typedef typename MatrixType::Scalar Value;
    std::ifstream input_stream;
    input_stream.open(path);
    if (!input_stream.good())
    {
        std::runtime_error("ifstream for " + path + " is not in  a valid state.");
    }
    std::string line;
    std::vector<Value> parsed_values;
    Index rows = 0;
    while (std::getline(input_stream, line))
    {
        std::stringstream line_stream(line);
        std::string cell;
        while (std::getline(line_stream, cell, sep))
        {
            parsed_values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Eigen::Map<
    const Eigen::Matrix<Value, MatrixType::RowsAtCompileTime,
                        MatrixType::ColsAtCompileTime, Eigen::RowMajor>>(
    parsed_values.data(), rows, parsed_values.size() / rows);
}

} // namespace io
} // namespace mimkl

#endif /* _IO_HPP_ */
