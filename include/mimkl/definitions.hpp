#ifndef INCLUDE_MIMKL_DEFINITIONS_HPP_
#define INCLUDE_MIMKL_DEFINITIONS_HPP_

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <map>

#define MATRIX(T) Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
#define ROW(T) Eigen::Matrix<T, 1, Eigen::Dynamic>
#define COLUMN(T) Eigen::Matrix<T, Eigen::Dynamic, 1>

namespace mimkl
{
namespace definitions
{

typedef unsigned int Index;
typedef std::multimap<std::string, Index> Indexing;
typedef std::map<Index, std::string> ReversedIndexing;
typedef MATRIX(double) Matrix;
typedef Eigen::Ref<const MATRIX(double)> RefConstMatrix;
typedef std::function<Matrix(const Matrix &, const Matrix &)> Kernel;
typedef COLUMN(double) Column;
typedef ROW(double) Row;
typedef Eigen::SparseMatrix<double> SparseMatrix;

} // namespace definitions
} // namespace mimkl

#endif /* INCLUDE_MIMKL_DEFINITIONS_HPP_ */
