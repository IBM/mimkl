#ifndef INCLUDE_MIMKL_DATA_STRUCTURES_DATA_FRAME_HPP_
#define INCLUDE_MIMKL_DATA_STRUCTURES_DATA_FRAME_HPP_

#include <Eigen/Core>
#include <mimkl/data_structures/utilities.hpp>
#include <mimkl/definitions.hpp>
#include <mimkl/linear_algebra.hpp>
#include <numeric>
#include <ostream>
#include <string>
#include <vector>

using mimkl::definitions::Index;
using mimkl::definitions::Indexing;

namespace mimkl
{
namespace data_structures
{

/*! The class DataFrame inherits from Eigen::MatrixXd and foremost provides
   methods to read and write data
   Also it provides label functionality, syntax is inspired by pandas.
*/
class DataFrame : public Eigen::MatrixXd
{

    private:
    Indexing index_;
    Indexing columns_;
    void build_indexing_selection(const std::vector<std::string> &,
                                  const Indexing &,
                                  u_int &,
                                  std::vector<int> &,
                                  Indexing &);
    DataFrame access_by_index(const std::vector<std::string> &);
    DataFrame access_by_columns(const std::vector<std::string> &);

    public:
    //! Default constructor
    /*! initialize all members of the base class by initialization-list
           instead of using the default constructor of MatrixXd ( which is
       protected )
     */
    DataFrame(void) : Eigen::MatrixXd() {}
    //! Copy constructor
    /*! to construct uninitialized DataFrame from Eigen expressions, w/o labels
     */
    template <typename OtherDerived>
    DataFrame(const Eigen::MatrixBase<OtherDerived> &other)
    : Eigen::MatrixXd(other),
      index_(indexing_from_vector(range(other.rows()))),
      columns_(indexing_from_vector(range(other.cols())))
    {
    }
    //! Copy constructor
    /*! to construct uninitialized DataFrame from Eigen expressions, giving the
     * labels
     */
    template <typename OtherDerived>
    DataFrame(const Eigen::MatrixBase<OtherDerived> &other,
              const Indexing &index,
              const Indexing &columns)
    : Eigen::MatrixXd(other), index_(index), columns_(columns)
    {
    }
    //! Assignment operator
    /*! This method allows you to assign Eigen expressions to existing DataFrame
     */
    template <typename OtherDerived>
    DataFrame &operator=(const Eigen::MatrixBase<OtherDerived> &other)
    {
        this->Eigen::MatrixXd::operator=(other);
        index_ = indexing_from_vector(range(other.rows()));
        columns_ = indexing_from_vector(range(other.cols()));
        return *this;
    }

    friend std::ostream &operator<<(std::ostream &, const DataFrame &);
    // access rows
    DataFrame loc(const std::string &);
    DataFrame loc(const std::vector<std::string> &);
    // access a column
    DataFrame operator[](const std::string &);
    DataFrame operator[](const std::vector<std::string> &);
    // static to be able to read a DataFrame without an instance:
    // DataFrame::read_csv()
    //	static DataFrame read_csv(const std::string &){ return nullptr;}; //
    // from csv get matrix and map
    // void to_csv();
};
} // namespace data_structures
} // namespace mimkl

#endif /* INCLUDE_MIMKL_DATA_STRUCTURES_DATA_FRAME_HPP_ */
