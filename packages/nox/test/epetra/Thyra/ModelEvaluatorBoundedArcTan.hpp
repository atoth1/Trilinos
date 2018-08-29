#ifndef NOX_THYRA_MODEL_EVALUATOR_BOUNDED_ARCTAN_DECL_HPP
#define NOX_THYRA_MODEL_EVALUATOR_BOUNDED_ARCTAN_DECL_HPP

#include "Thyra_StateFuncModelEvaluatorBase.hpp"

class Epetra_Comm;
class Epetra_Map;
template<class Scalar> class ModelEvaluatorBoundedArcTan;

/** \brief Nonmember constuctor.
 *
 * \relates ModelEvaluatorBoundedArcTan
 */
template<class Scalar>
Teuchos::RCP<ModelEvaluatorBoundedArcTan<Scalar> >
modelEvaluatorBoundedArcTan(const Teuchos::RCP<const Epetra_Comm>& comm);

/** \brief Simple ModelEvaluator for f(x) = 0.
 *
 * The equations modeled are:

 \verbatim

    f(x) = arctan(x-1), x >= 0
           nan,         x < 0.

 \endverbatim

 * This is a contrived example to test line searches in the case where
 * residual evaluations may result in a nan.
 */
template<class Scalar>
class ModelEvaluatorBoundedArcTan
  : public Thyra::StateFuncModelEvaluatorBase<Scalar>
{
public:

  ModelEvaluatorBoundedArcTan(const Teuchos::RCP<const Epetra_Comm>& comm);

  /** \name Initializers/Accessors */
  //@{

  void set_W_factory(const Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<Scalar> >& W_factory);

  //@}

  /** \name Public functions overridden from ModelEvaulator. */
  //@{

  /** \brief . */
  Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> > get_x_space() const;
  /** \brief . */
  Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> > get_f_space() const;
  /** \brief . */
  Thyra::ModelEvaluatorBase::InArgs<Scalar> getNominalValues() const;
  /** \brief . */
  Teuchos::RCP< Thyra::LinearOpBase<Scalar> > create_W_op() const;
  /** \brief . */
  Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<Scalar> > get_W_factory() const;
  /** \brief . */
  Thyra::ModelEvaluatorBase::InArgs<Scalar> createInArgs() const;
  //@}

private:

  /** \name Private functions overridden from ModelEvaulatorDefaultBase. */
  //@{

  /** \brief . */
  Thyra::ModelEvaluatorBase::OutArgs<Scalar> createOutArgsImpl() const;

  /** \brief . */
  void evalModelImpl(
    const Thyra::ModelEvaluatorBase::InArgs<Scalar> &inArgs,
    const Thyra::ModelEvaluatorBase::OutArgs<Scalar> &outArgs
    ) const;

  //@}

private: // data members

  Teuchos::RCP<const Epetra_Comm>  epetra_comm_;

  Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> > x_space_;
  Teuchos::RCP<const Epetra_Map>   x_epetra_map_;

  Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> > f_space_;
  Teuchos::RCP<const Epetra_Map>   f_epetra_map_;

  Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<Scalar> > W_factory_;

  Thyra::ModelEvaluatorBase::InArgs<Scalar> nominalValues_;
  Teuchos::RCP< Thyra::VectorBase<Scalar> > x0_;
  Thyra::ModelEvaluatorBase::InArgs<Scalar> prototypeInArgs_;
  Thyra::ModelEvaluatorBase::OutArgs<Scalar> prototypeOutArgs_;

};

#include "ModelEvaluatorBoundedArcTan_def.hpp"

#endif // NOX_THYRA_MODEL_EVALUATOR_BOUNDED_ARCTAN_DECL_HPP
