// @HEADER
//
// ***********************************************************************
//
//             Xpetra: A linear algebra interface package
//                  Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact
//                    Jonathan Hu       (jhu@sandia.gov)
//                    Andrey Prokopenko (aprokop@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#ifndef XPETRA_EPETRAMULTIVECTOR_HPP
#define XPETRA_EPETRAMULTIVECTOR_HPP

/* this file is automatically generated - do not edit (see script/epetra.py) */

#ifdef HAVE_XPETRA_KOKKOS_REFACTOR
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#endif


#include "Xpetra_EpetraConfigDefs.hpp"

#include "Xpetra_MultiVector.hpp"
#include "Xpetra_Vector.hpp"

#include "Xpetra_EpetraMap.hpp"
#include "Xpetra_EpetraExport.hpp"
#include "Xpetra_Utils.hpp"
#include "Xpetra_EpetraUtils.hpp"
#include "Xpetra_EpetraImport.hpp"
#include "Xpetra_Exceptions.hpp"
#include "Epetra_SerialComm.h"

#include <Epetra_MultiVector.h>

namespace Xpetra {

  // TODO: move that elsewhere
  template<class GlobalOrdinal>
  const Epetra_MultiVector &          toEpetra(const MultiVector<double,int,GlobalOrdinal> &);
  template<class GlobalOrdinal>
  Epetra_MultiVector &                toEpetra(MultiVector<double, int,GlobalOrdinal> &);
  template<class GlobalOrdinal>
  RCP<MultiVector<double, int, GlobalOrdinal> > toXpetra(RCP<Epetra_MultiVector> vec);

  // #ifndef DOXYGEN_SHOULD_SKIP_THIS
  //   // forward declaration of EpetraVectorT, needed to prevent circular inclusions
  //   template<class S, class LO, class GO, class N> class EpetraVectorT;
  // #endif

  template<class EpetraGlobalOrdinal>
  class EpetraMultiVectorT
    : public virtual MultiVector<double, int, EpetraGlobalOrdinal>
  {
    typedef double Scalar;
    typedef int LocalOrdinal;
    typedef EpetraGlobalOrdinal GlobalOrdinal;
    typedef typename MultiVector<double, int, GlobalOrdinal>::node_type Node;

  public:

    //! @name Constructor/Destructor Methods
    //@{

    //! Basic MultiVector constuctor.
    EpetraMultiVectorT(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &map, size_t NumVectors, bool zeroOut=true)
      : vec_(Teuchos::rcp(new Epetra_MultiVector(toEpetra(map), Teuchos::as<int>(NumVectors), zeroOut))) { }

    //! MultiVector copy constructor.
    EpetraMultiVectorT(const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &source)
      : vec_(Teuchos::rcp(new Epetra_MultiVector(toEpetra(source)))) { }

    //! Set multi-vector values from array of pointers using Teuchos memory management classes. (copy).
    EpetraMultiVectorT(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &map, const Teuchos::ArrayView< const Teuchos::ArrayView< const Scalar > > &ArrayOfPtrs, size_t NumVectors);

    //! MultiVector destructor.
    virtual ~EpetraMultiVectorT() {}

    //@}

    //! @name Post-construction modification routines
    //@{

    //! Replace value, using global (row) index.
    void replaceGlobalValue(GlobalOrdinal globalRow, size_t vectorIndex, const Scalar &value) { XPETRA_MONITOR("EpetraMultiVectorT::replaceGlobalValue"); vec_->ReplaceGlobalValue(globalRow, Teuchos::as<int>(vectorIndex), value); }

    //! Add value to existing value, using global (row) index.
    void sumIntoGlobalValue(GlobalOrdinal globalRow, size_t vectorIndex, const Scalar &value) { XPETRA_MONITOR("EpetraMultiVectorT::sumIntoGlobalValue"); vec_->SumIntoGlobalValue(globalRow, Teuchos::as<int>(vectorIndex), value); }

    //! Replace value, using local (row) index.
    void replaceLocalValue(LocalOrdinal myRow, size_t vectorIndex, const Scalar &value) { XPETRA_MONITOR("EpetraMultiVectorT::replaceLocalValue"); vec_->ReplaceMyValue(myRow, Teuchos::as<int>(vectorIndex), value); }

    //! Add value to existing value, using local (row) index.
    void sumIntoLocalValue(LocalOrdinal myRow, size_t vectorIndex, const Scalar &value) { XPETRA_MONITOR("EpetraMultiVectorT::sumIntoLocalValue"); vec_->SumIntoMyValue(myRow, Teuchos::as<int>(vectorIndex), value); }

    //! Set all values in the multivector with the given value.
    void putScalar(const Scalar &value) { XPETRA_MONITOR("EpetraMultiVectorT::putScalar"); vec_->PutScalar(value); }

    //@}

    //! @name Data copy and view methods
    //@{

    //! Return a Vector which is a const view of column j.
    Teuchos::RCP< const Vector< double, int, GlobalOrdinal, Node > > getVector(size_t j) const;

    //! Return a Vector which is a nonconst view of column j.
    Teuchos::RCP< Vector< double, int, GlobalOrdinal, Node > > getVectorNonConst(size_t j);

    //! Const view of the local values in a particular vector of this multivector.
    Teuchos::ArrayRCP< const Scalar > getData(size_t j) const;

    //! View of the local values in a particular vector of this multivector.
    Teuchos::ArrayRCP< Scalar > getDataNonConst(size_t j);

    //@}

    //! @name Mathematical methods
    //@{

    //! Compute the dot product of each corresponding pair of vectors (columns) in A and B.
    void dot(const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A, const Teuchos::ArrayView< Scalar > &dots) const;

    //! Put element-wise absolute values of input Multi-vector in target: A = abs(this).
    void abs(const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A) { XPETRA_MONITOR("EpetraMultiVectorT::abs"); vec_->Abs(toEpetra(A)); }

    //! Put element-wise reciprocal values of input Multi-vector in target, this(i,j) = 1/A(i,j).
    void reciprocal(const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A) { XPETRA_MONITOR("EpetraMultiVectorT::reciprocal"); vec_->Reciprocal(toEpetra(A)); }

    //! Scale in place: this = alpha*this.
    void scale(const Scalar &alpha) { XPETRA_MONITOR("EpetraMultiVectorT::scale"); vec_->Scale(alpha); }

    //! Scale the current values of a multi-vector, this[j] = alpha[j]*this[j].
    void scale (Teuchos::ArrayView< const Scalar > alpha) {
      XPETRA_MONITOR("EpetraMultiVectorT::scale");
      // Epetra, unlike Tpetra, doesn't implement this version of
      // scale().  Deal with this by scaling one column at a time.
      const size_t numVecs = this->getNumVectors ();
      for (size_t j = 0; j < numVecs; ++j) {
        vec_->Scale (alpha[j]);
      }
    }

    //! Update: this = beta*this + alpha*A.
    void update(const Scalar &alpha, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A, const Scalar &beta) { XPETRA_MONITOR("EpetraMultiVectorT::update"); vec_->Update(alpha, toEpetra(A), beta); }

    //! Update: this = gamma*this + alpha*A + beta*B.
    void update(const Scalar &alpha, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A, const Scalar &beta, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &B, const Scalar &gamma) { XPETRA_MONITOR("EpetraMultiVectorT::update"); vec_->Update(alpha, toEpetra(A), beta, toEpetra(B), gamma); }

    //! Compute 1-norm of each vector in multi-vector.
    void norm1(const Teuchos::ArrayView< Teuchos::ScalarTraits< Scalar >::magnitudeType > &norms) const;

    //!
    void norm2(const Teuchos::ArrayView< Teuchos::ScalarTraits< Scalar >::magnitudeType > &norms) const;

    //! Compute Inf-norm of each vector in multi-vector.
    void normInf(const Teuchos::ArrayView< Teuchos::ScalarTraits< Scalar >::magnitudeType > &norms) const;

    //! Compute mean (average) value of each vector in multi-vector. The outcome of this routine is undefined for non-floating point scalar types (e.g., int).
    void meanValue(const Teuchos::ArrayView< Scalar > &means) const;

    //! Matrix-matrix multiplication: this = beta*this + alpha*op(A)*op(B).
    void multiply(Teuchos::ETransp transA, Teuchos::ETransp transB, const Scalar &alpha, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &B, const Scalar &beta) { XPETRA_MONITOR("EpetraMultiVectorT::multiply"); vec_->Multiply(toEpetra(transA), toEpetra(transB), alpha, toEpetra(A), toEpetra(B), beta); }

    //! Multiply a Vector A elementwise by a MultiVector B.
    void elementWiseMultiply(Scalar scalarAB, const Vector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &A, const MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node > &B, Scalar scalarThis) { XPETRA_MONITOR("EpetraMultiVectorT::elementWiseMultiply"); vec_->Multiply(scalarAB, toEpetra(A), toEpetra(B), scalarThis); }

    //@}

    //! @name Attribute access functions
    //@{

    //! Number of columns in the multivector.
    size_t getNumVectors() const { XPETRA_MONITOR("EpetraMultiVectorT::getNumVectors"); return vec_->NumVectors(); }

    //! Local number of rows on the calling process.
    size_t getLocalLength() const { XPETRA_MONITOR("EpetraMultiVectorT::getLocalLength"); return vec_->MyLength(); }

    //! Global number of rows in the multivector.
    global_size_t getGlobalLength() const { XPETRA_MONITOR("EpetraMultiVectorT::getGlobalLength"); return vec_->GlobalLength64(); }

    //@}

    //! @name Overridden from Teuchos::Describable
    //@{

    //! A simple one-line description of this object.
    std::string description() const;

    //! Print the object with the given verbosity level to a FancyOStream.
    void describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel=Teuchos::Describable::verbLevel_default) const;

    //@}

    //! Set multi-vector values to random numbers.
    void randomize(bool bUseXpetraImplementation = false) {
      XPETRA_MONITOR("EpetraMultiVectorT::randomize");

      if (bUseXpetraImplementation)
        Xpetra::MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node >::Xpetra_randomize();
      else
        vec_->Random();
    }

    //! Implements DistObject interface
    //{@

    //! Access function for the Tpetra::Map this DistObject was constructed with.
    Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > getMap() const { XPETRA_MONITOR("EpetraMultiVectorT::getMap"); return toXpetra<GlobalOrdinal>(vec_->Map()); }

    //! Import.
    void doImport(const DistObject<Scalar, LocalOrdinal, GlobalOrdinal, Node> &source, const Import< LocalOrdinal, GlobalOrdinal, Node > &importer, CombineMode CM);

    //! Export.
    void doExport(const DistObject<Scalar, LocalOrdinal, GlobalOrdinal, Node> &dest, const Import< LocalOrdinal, GlobalOrdinal, Node >& importer, CombineMode CM);

    //! Import (using an Exporter).
    void doImport(const DistObject<Scalar, LocalOrdinal, GlobalOrdinal, Node> &source, const Export< LocalOrdinal, GlobalOrdinal, Node >& exporter, CombineMode CM);

    //! Export (using an Importer).
    void doExport(const DistObject<Scalar, LocalOrdinal, GlobalOrdinal, Node> &dest, const Export< LocalOrdinal, GlobalOrdinal, Node >& exporter, CombineMode CM);

    //! Replace the underlying Map in place.
    void replaceMap(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &map);

    //@}

    //! @name Xpetra specific
    //@{

    //! EpetraMultiVectorT constructor to wrap a Epetra_MultiVector object
    EpetraMultiVectorT(const RCP<Epetra_MultiVector> &vec) : vec_(vec) { } //TODO removed const

    //! Get the underlying Epetra multivector
    RCP<Epetra_MultiVector> getEpetra_MultiVector() const { return vec_; }

    //! Set seed for Random function.
    void setSeed(unsigned int seed) {
      XPETRA_MONITOR("EpetraMultiVectorT::seedrandom");

      Teuchos::ScalarTraits< Scalar >::seedrandom(seed);
      vec_->SetSeed(seed);
    }

#ifdef HAVE_XPETRA_KOKKOS_REFACTOR

    typedef typename Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>::dual_view_type dual_view_type;

    /// \brief Return an unmanaged non-const view of the local data on a specific device.
    /// \tparam TargetDeviceType The Kokkos Device type whose data to return.
    ///
    /// \warning DO NOT USE THIS FUNCTION! There is no reason why you are working directly
    ///          with the Xpetra::EpetraMultiVector object. To write a code which is independent
    ///          from the underlying linear algebra package you should always use the abstract class,
    ///          i.e. Xpetra::MultiVector!
    ///
    /// \warning Be aware that the view on the multivector data is non-persisting, i.e.
    ///          only valid as long as the multivector does not run of scope!
    template<class TargetDeviceType>
    typename Kokkos::Impl::if_c<
      Kokkos::Impl::is_same<
        typename dual_view_type::t_dev_um::execution_space::memory_space,
        typename TargetDeviceType::memory_space>::value,
        typename dual_view_type::t_dev_um,
        typename dual_view_type::t_host_um>::type
    getLocalView () const {
      return this->MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node >::template getLocalView<TargetDeviceType>();
    }

    typename dual_view_type::t_host_um getHostLocalView () const {
      typedef Kokkos::View< typename dual_view_type::t_host::data_type ,
                    Kokkos::LayoutLeft,
                    typename dual_view_type::t_host::device_type ,
                    Kokkos::MemoryUnmanaged> epetra_view_type;

      // access Epetra multivector data
      double* data = NULL;
      int myLDA;
      vec_->ExtractView(&data, &myLDA);
      int localLength = vec_->MyLength();
      int numVectors  = getNumVectors();

      // create view
      epetra_view_type test = epetra_view_type(data, localLength, numVectors);
      typename dual_view_type::t_host_um ret = subview(test, Kokkos::ALL(), Kokkos::ALL());

      return ret;
    }

    typename dual_view_type::t_dev_um getDeviceLocalView() const {
      throw std::runtime_error("Epetra does not support device views!");
      typename dual_view_type::t_dev_um ret;
      return ret; // make compiler happy
    }

#endif

    //@}

  protected:
    /// \brief Implementation of the assignment operator (operator=);
    ///   does a deep copy.
    virtual void
    assign (const MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& rhs);

  private:
    //! The Epetra_MultiVector which this class wraps.
    RCP< Epetra_MultiVector > vec_;

  }; // EpetraMultiVectorT class

#ifndef XPETRA_EPETRA_NO_32BIT_GLOBAL_INDICES
  typedef EpetraMultiVectorT<int> EpetraMultiVector;
#endif

#ifndef XPETRA_EPETRA_NO_64BIT_GLOBAL_INDICES
  typedef EpetraMultiVectorT<long long> EpetraMultiVector64;
#endif

// Moving here from cpp since some compilers don't have public visibility of virtual thunks.
// https://software.sandia.gov/bugzilla/show_bug.cgi?id=6232

  template<class> class EpetraVectorT;

  template<class EpetraGlobalOrdinal>
  EpetraMultiVectorT<EpetraGlobalOrdinal>::EpetraMultiVectorT(const Teuchos::RCP< const Map< LocalOrdinal, GlobalOrdinal, Node > > &map, const Teuchos::ArrayView< const Teuchos::ArrayView< const Scalar > > &ArrayOfPtrs, size_t NumVectors) {
    //TODO: input argument 'NumVectors' is not necessary in both Xpetra and Tpetra interface. Should it be removed?

    const std::string tfecfFuncName("MultiVector(ArrayOfPtrs)");
    TEUCHOS_TEST_FOR_EXCEPTION_CLASS_FUNC(NumVectors < 1 || NumVectors != Teuchos::as<size_t>(ArrayOfPtrs.size()), std::runtime_error,
                                          ": ArrayOfPtrs.size() must be strictly positive and as large as ArrayOfPtrs.");

#ifdef HAVE_XPETRA_DEBUG
    // This cannot be tested by Epetra itself
    {
      size_t localLength = map->getNodeNumElements();
      for(int j=0; j<ArrayOfPtrs.size(); j++) {
        TEUCHOS_TEST_FOR_EXCEPTION_CLASS_FUNC(Teuchos::as<size_t>(ArrayOfPtrs[j].size()) != localLength, std::runtime_error,
                                              ": ArrayOfPtrs[" << j << "].size() (== " << ArrayOfPtrs[j].size() <<
                                              ") is not equal to getLocalLength() (== " << localLength);

      }
    }
#endif

    // Convert Teuchos::ArrayView< const Teuchos::ArrayView< const Scalar > > to double**
    Array<const double*> arrayOfRawPtrs(ArrayOfPtrs.size());
    for(int i=0; i<ArrayOfPtrs.size(); i++) {
      arrayOfRawPtrs[i] = ArrayOfPtrs[i].getRawPtr();
    }
    double** rawArrayOfRawPtrs = const_cast<double**>(arrayOfRawPtrs.getRawPtr()); // This const_cast should be fine, because Epetra_DataAccess=Copy.

    vec_ = Teuchos::rcp(new Epetra_MultiVector(Copy, toEpetra(map), rawArrayOfRawPtrs, NumVectors));
  }


  template<class EpetraGlobalOrdinal>
  Teuchos::RCP< const Vector< double, int, EpetraGlobalOrdinal, typename EpetraMultiVectorT<EpetraGlobalOrdinal>::Node > > EpetraMultiVectorT<EpetraGlobalOrdinal>::getVector(size_t j) const {
    XPETRA_MONITOR("EpetraMultiVectorT::getVector");
    return rcp(new EpetraVectorT<GlobalOrdinal>(vec_, j)); // See constructor EpetraVectorT(const RCP<EpetraMultiVectorT> &mv, size_t j) for more info
  }

  template<class EpetraGlobalOrdinal>
  Teuchos::RCP< Vector< double, int, EpetraGlobalOrdinal, typename EpetraMultiVectorT<EpetraGlobalOrdinal>::Node > > EpetraMultiVectorT<EpetraGlobalOrdinal>::getVectorNonConst(size_t j) {
    XPETRA_MONITOR("EpetraMultiVectorT::getVector");
    return rcp(new EpetraVectorT<GlobalOrdinal>(vec_, j)); // See constructor EpetraVectorT(const RCP<EpetraMultiVectorT> &mv, size_t j) for more info
  }

  template<class EpetraGlobalOrdinal>
  Teuchos::ArrayRCP<const double> EpetraMultiVectorT<EpetraGlobalOrdinal>::getData(size_t j) const {
    XPETRA_MONITOR("EpetraMultiVectorT::getData");

    double ** arrayOfPointers;

    vec_->ExtractView(&arrayOfPointers);

    double * data = arrayOfPointers[j];
    int localLength = vec_->MyLength();

    return ArrayRCP<double>(data, 0, localLength, false); // no ownership
  }

  template<class EpetraGlobalOrdinal>
  Teuchos::ArrayRCP<double> EpetraMultiVectorT<EpetraGlobalOrdinal>::getDataNonConst(size_t j) {
    XPETRA_MONITOR("EpetraMultiVectorT::getDataNonConst");

    double ** arrayOfPointers;

    vec_->ExtractView(&arrayOfPointers);

    double * data = arrayOfPointers[j];
    int localLength = vec_->MyLength();

    return ArrayRCP<double>(data, 0, localLength, false); // no ownership
  }

  template<class EpetraGlobalOrdinal>
  void EpetraMultiVectorT<EpetraGlobalOrdinal>::dot(const MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> &A, const Teuchos::ArrayView<Scalar> &dots) const {
    XPETRA_MONITOR("EpetraMultiVectorT::dot");

    XPETRA_DYNAMIC_CAST(const EpetraMultiVectorT, A, eA, "This Xpetra::EpetraMultiVectorT method only accept Xpetra::EpetraMultiVectorT as input arguments.");
    vec_->Dot(*eA.getEpetra_MultiVector(), dots.getRawPtr());
  }

  template<class EpetraGlobalOrdinal>
  void EpetraMultiVectorT<EpetraGlobalOrdinal>::norm1(const Teuchos::ArrayView< Teuchos::ScalarTraits< Scalar >::magnitudeType > &norms) const { XPETRA_MONITOR("EpetraMultiVectorT::norm1"); vec_->Norm1(norms.getRawPtr()); }

  template<class EpetraGlobalOrdinal>
  void EpetraMultiVectorT<EpetraGlobalOrdinal>::norm2(const Teuchos::ArrayView< Teuchos::ScalarTraits< Scalar >::magnitudeType > &norms) const { XPETRA_MONITOR("EpetraMultiVectorT::norm2"); vec_->Norm2(norms.getRawPtr()); }

  template<class EpetraGlobalOrdinal>
  void EpetraMultiVectorT<EpetraGlobalOrdinal>::normInf(const Teuchos::ArrayView< Teuchos::ScalarTraits< Scalar >::magnitudeType > &norms) const { XPETRA_MONITOR("EpetraMultiVectorT::normInf"); vec_->NormInf(norms.getRawPtr()); }

  template<class EpetraGlobalOrdinal>
  void EpetraMultiVectorT<EpetraGlobalOrdinal>::meanValue(const Teuchos::ArrayView<double> &means) const { XPETRA_MONITOR("EpetraMultiVectorT::meanValue"); vec_->MeanValue(means.getRawPtr()); } //TODO: modify ArrayView size ??

  template<class EpetraGlobalOrdinal>
  std::string EpetraMultiVectorT<EpetraGlobalOrdinal>::description() const {
    XPETRA_MONITOR("EpetraMultiVectorT::description");
    TEUCHOS_TEST_FOR_EXCEPTION(1, Xpetra::Exceptions::NotImplemented, "TODO");
    return "TODO";
  }

  template<class EpetraGlobalOrdinal>
  void EpetraMultiVectorT<EpetraGlobalOrdinal>::describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel) const {
    XPETRA_MONITOR("EpetraMultiVectorT::describe");
    vec_->Print(out);
  }

  template<class EpetraGlobalOrdinal>
  void EpetraMultiVectorT<EpetraGlobalOrdinal>::doImport(const DistObject<double, int, GlobalOrdinal, Node> &source, const Import<int, GlobalOrdinal, Node> &importer, CombineMode CM) {
    XPETRA_MONITOR("EpetraMultiVectorT::doImport");

    XPETRA_DYNAMIC_CAST(const EpetraMultiVectorT<GlobalOrdinal>, source, tSource, "Xpetra::EpetraMultiVectorT::doImport only accept Xpetra::EpetraMultiVectorT as input arguments.");
    XPETRA_DYNAMIC_CAST(const EpetraImportT<GlobalOrdinal>, importer, tImporter, "Xpetra::EpetraMultiVectorT::doImport only accept Xpetra::EpetraImportT as input arguments.");

    RCP<Epetra_MultiVector> v = tSource.getEpetra_MultiVector();
    int err = this->getEpetra_MultiVector()->Import(*v, *tImporter.getEpetra_Import(), toEpetra(CM));
    TEUCHOS_TEST_FOR_EXCEPTION(err != 0, std::runtime_error, "Catch error code returned by Epetra.");
  }

  template<class EpetraGlobalOrdinal>
  void EpetraMultiVectorT<EpetraGlobalOrdinal>::doExport(const DistObject<double, int, GlobalOrdinal, Node> &dest, const Import<int, GlobalOrdinal, Node>& importer, CombineMode CM) {
    XPETRA_MONITOR("EpetraMultiVectorT::doExport");

    XPETRA_DYNAMIC_CAST(const EpetraMultiVectorT<GlobalOrdinal>, dest, tDest, "Xpetra::EpetraMultiVectorT::doImport only accept Xpetra::EpetraMultiVectorT as input arguments.");
    XPETRA_DYNAMIC_CAST(const EpetraImportT<GlobalOrdinal>, importer, tImporter, "Xpetra::EpetraMultiVectorT::doImport only accept Xpetra::EpetraImportT as input arguments.");

    RCP<Epetra_MultiVector> v = tDest.getEpetra_MultiVector();
    int err = this->getEpetra_MultiVector()->Export(*v, *tImporter.getEpetra_Import(), toEpetra(CM));
    TEUCHOS_TEST_FOR_EXCEPTION(err != 0, std::runtime_error, "Catch error code returned by Epetra.");
  }

  template<class EpetraGlobalOrdinal>
  void EpetraMultiVectorT<EpetraGlobalOrdinal>::doImport(const DistObject<double,int,GlobalOrdinal,Node> &source, const Export<int, GlobalOrdinal, Node>& exporter, CombineMode CM) {
    XPETRA_MONITOR("EpetraMultiVectorT::doImport");

    XPETRA_DYNAMIC_CAST(const EpetraMultiVectorT<GlobalOrdinal>, source, tSource, "Xpetra::EpetraMultiVectorT::doImport only accept Xpetra::EpetraMultiVectorT as input arguments.");
    XPETRA_DYNAMIC_CAST(const EpetraExportT<GlobalOrdinal>, exporter, tExporter, "Xpetra::EpetraMultiVectorT::doImport only accept Xpetra::EpetraImportT as input arguments.");

    RCP<Epetra_MultiVector> v = tSource.getEpetra_MultiVector();
    int err = this->getEpetra_MultiVector()->Import(*v, *tExporter.getEpetra_Export(), toEpetra(CM));
    TEUCHOS_TEST_FOR_EXCEPTION(err != 0, std::runtime_error, "Catch error code returned by Epetra.");
  }

  template<class EpetraGlobalOrdinal>
  void EpetraMultiVectorT<EpetraGlobalOrdinal>::doExport(const DistObject<double, int, GlobalOrdinal, Node> &dest, const Export<int, GlobalOrdinal, Node>& exporter, CombineMode CM) {
    XPETRA_MONITOR("EpetraMultiVectorT::doExport");

    XPETRA_DYNAMIC_CAST(const EpetraMultiVectorT<GlobalOrdinal>, dest, tDest, "Xpetra::EpetraMultiVectorT::doImport only accept Xpetra::EpetraMultiVectorT as input arguments.");
    XPETRA_DYNAMIC_CAST(const EpetraExportT<GlobalOrdinal>, exporter, tExporter, "Xpetra::EpetraMultiVectorT::doImport only accept Xpetra::EpetraImportT as input arguments.");

    RCP<Epetra_MultiVector> v = tDest.getEpetra_MultiVector();
    int err = this->getEpetra_MultiVector()->Export(*v, *tExporter.getEpetra_Export(), toEpetra(CM));
    TEUCHOS_TEST_FOR_EXCEPTION(err != 0, std::runtime_error, "Catch error code returned by Epetra.");
  }

  template<class EpetraGlobalOrdinal>
  void EpetraMultiVectorT<EpetraGlobalOrdinal>::replaceMap(const RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> >& map) {
    int err = 0;
    if (!map.is_null()) {
      err = this->getEpetra_MultiVector()->ReplaceMap(toEpetra(map));

    } else {
      // Replace map with a dummy map to avoid potential hangs later
      Epetra_SerialComm SComm;
      Epetra_Map NewMap((EpetraGlobalOrdinal) vec_->MyLength(), (EpetraGlobalOrdinal) vec_->Map().IndexBase64(), SComm);
      err = this->getEpetra_MultiVector()->ReplaceMap(NewMap);
    }
    TEUCHOS_TEST_FOR_EXCEPTION(err != 0, std::runtime_error, "Catch error code returned by Epetra.");
  }

  template<class EpetraGlobalOrdinal>
  void EpetraMultiVectorT<EpetraGlobalOrdinal>::
  assign (const MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& rhs)
  {
    typedef EpetraMultiVectorT this_type;
    const this_type* rhsPtr = dynamic_cast<const this_type*> (&rhs);
    TEUCHOS_TEST_FOR_EXCEPTION(
      rhsPtr == NULL, std::invalid_argument, "Xpetra::MultiVector::operator=: "
      "The left-hand side (LHS) of the assignment has a different type than "
      "the right-hand side (RHS).  The LHS has type Xpetra::EpetraMultiVectorT "
      "(which means it wraps an Epetra_MultiVector), but the RHS has some "
      "other type.  This probably means that the RHS wraps a Tpetra::Multi"
      "Vector.  Xpetra::MultiVector does not currently implement assignment "
      "from a Tpetra object to an Epetra object, though this could be added "
      "with sufficient interest.");

    RCP<const Epetra_MultiVector> rhsImpl = rhsPtr->getEpetra_MultiVector ();
    RCP<Epetra_MultiVector> lhsImpl = this->getEpetra_MultiVector ();

    TEUCHOS_TEST_FOR_EXCEPTION(
      rhsImpl.is_null (), std::logic_error, "Xpetra::MultiVector::operator= "
      "(in Xpetra::EpetraMultiVectorT::assign): *this (the right-hand side of "
      "the assignment) has a null RCP<Epetra_MultiVector> inside.  Please "
      "report this bug to the Xpetra developers.");
    TEUCHOS_TEST_FOR_EXCEPTION(
      lhsImpl.is_null (), std::logic_error, "Xpetra::MultiVector::operator= "
      "(in Xpetra::EpetraMultiVectorT::assign): The left-hand side of the "
      "assignment has a null RCP<Epetra_MultiVector> inside.  Please report "
      "this bug to the Xpetra developers.");

    // Epetra_MultiVector's assignment operator does a deep copy.
    *lhsImpl = *rhsImpl;
  }


} // Xpetra namespace

#include "Xpetra_EpetraVector.hpp" // to avoid incomplete type instantiated above in out-of-body functions.

#endif // XPETRA_EPETRAMULTIVECTOR_HPP
