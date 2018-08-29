#ifndef NOX_THYRA_MODEL_EVALUATOR_BOUNDED_ARCTAN_DEF_HPP
#define NOX_THYRA_MODEL_EVALUATOR_BOUNDED_ARCTAN_DEF_HPP

#include "Epetra_Comm.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Vector.h"

#include "Teuchos_as.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Thyra_VectorStdOps.hpp"
#include "Thyra_EpetraThyraWrappers.hpp"
#include "Thyra_get_Epetra_Operator.hpp"

#include <cmath>

template<class Scalar>
Teuchos::RCP<ModelEvaluatorBoundedArcTan<Scalar> >
modelEvaluatorBoundedArcTan(const Teuchos::RCP<const Epetra_Comm>& comm)
{
  return Teuchos::rcp(new ModelEvaluatorBoundedArcTan<Scalar>(comm));
}

template<class Scalar>
ModelEvaluatorBoundedArcTan<Scalar>::ModelEvaluatorBoundedArcTan(const Teuchos::RCP<const Epetra_Comm>& comm)
  : epetra_comm_(comm)
{
  TEUCHOS_ASSERT(nonnull(epetra_comm_));

  x_epetra_map_ = Teuchos::rcp(new Epetra_Map(1, 0, *epetra_comm_));
  x_space_ = Thyra::create_VectorSpace(x_epetra_map_);

  f_epetra_map_ = x_epetra_map_;
  f_space_ = x_space_;

  x0_ = Thyra::createMember(x_space_);
  Thyra::V_S(x0_.ptr(), Teuchos::as<Scalar>(5.0));

  typedef Thyra::ModelEvaluatorBase MEB;
  MEB::InArgsSetup<Scalar> inArgs;
  inArgs.setModelEvalDescription(this->description());
  inArgs.setSupports(MEB::IN_ARG_x);
  prototypeInArgs_ = inArgs;

  MEB::OutArgsSetup<Scalar> outArgs;
  outArgs.setModelEvalDescription(this->description());
  outArgs.setSupports(MEB::OUT_ARG_f);
  outArgs.setSupports(MEB::OUT_ARG_W_op);
  prototypeOutArgs_ = outArgs;

  nominalValues_ = inArgs;
  nominalValues_.set_x(x0_);
}

// Initializers/Accessors

template<class Scalar>
void ModelEvaluatorBoundedArcTan<Scalar>::
set_W_factory(const Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<Scalar> >& W_factory)
{
  W_factory_ = W_factory;
}

// Public functions overridden from ModelEvaulator

template<class Scalar>
Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> >
ModelEvaluatorBoundedArcTan<Scalar>::get_x_space() const
{
  return x_space_;
}

template<class Scalar>
Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> >
ModelEvaluatorBoundedArcTan<Scalar>::get_f_space() const
{
  return f_space_;
}

template<class Scalar>
Thyra::ModelEvaluatorBase::InArgs<Scalar>
ModelEvaluatorBoundedArcTan<Scalar>::getNominalValues() const
{
  return nominalValues_;
}

template<class Scalar>
Teuchos::RCP<Thyra::LinearOpBase<Scalar> >
ModelEvaluatorBoundedArcTan<Scalar>::create_W_op() const
{
  Teuchos::RCP<Epetra_CrsMatrix> W_epetra =
    Teuchos::rcp(new Epetra_CrsMatrix(::Copy,*f_epetra_map_, 1, true));
  int col = 0;
  double value = 0.0;
  W_epetra->InsertGlobalValues(0, 1, &value, &col);
  W_epetra->FillComplete();

  return Thyra::nonconstEpetraLinearOp(W_epetra);
}

template<class Scalar>
Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<Scalar> >
ModelEvaluatorBoundedArcTan<Scalar>::get_W_factory() const
{
  return W_factory_;
}

template<class Scalar>
Thyra::ModelEvaluatorBase::InArgs<Scalar>
ModelEvaluatorBoundedArcTan<Scalar>::createInArgs() const
{
  return prototypeInArgs_;
}

// Private functions overridden from ModelEvaulatorDefaultBase

template<class Scalar>
Thyra::ModelEvaluatorBase::OutArgs<Scalar>
ModelEvaluatorBoundedArcTan<Scalar>::createOutArgsImpl() const
{
  return prototypeOutArgs_;
}

template<class Scalar>
void ModelEvaluatorBoundedArcTan<Scalar>::evalModelImpl(
  const Thyra::ModelEvaluatorBase::InArgs<Scalar> &inArgs,
  const Thyra::ModelEvaluatorBase::OutArgs<Scalar> &outArgs
  ) const
{
  TEUCHOS_ASSERT(nonnull(inArgs.get_x()));
  typedef Teuchos::ScalarTraits<Scalar> ST;

  const Teuchos::RCP<const Thyra::VectorBase<Scalar> > x_in = inArgs.get_x();
  const Teuchos::RCP<const Epetra_Vector> x = Thyra::get_Epetra_Vector(*x_epetra_map_, x_in);

  const Teuchos::RCP<Thyra::VectorBase<Scalar> > f_out = outArgs.get_f();
  const Teuchos::RCP<Thyra::LinearOpBase<Scalar> > W_out = outArgs.get_W_op();

  if (nonnull(f_out)) {
    NOX_FUNC_TIME_MONITOR("ModelEvaluatorBoundedArcTan::eval f_out");
    const Teuchos::RCP<Epetra_Vector> f = Thyra::get_Epetra_Vector(*f_epetra_map_, f_out);
    (*f)[0] = (*x)[0] < -5.0 ? ST::nan() : std::atan((*x)[0] - 1.0);
  }

  if (nonnull(W_out)) {
    NOX_FUNC_TIME_MONITOR("ModelEvaluatorBoundedArcTan::eval W_op_out");
    Teuchos::RCP<Epetra_Operator> W_epetra = Thyra::get_Epetra_Operator(*W_out);
    Teuchos::RCP<Epetra_CrsMatrix> W_epetracrs = Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(W_epetra);

    TEUCHOS_ASSERT(nonnull(W_epetracrs));

    int index = 0;
    double value = (*x)[0] < -5.0 ? 0.0 : 1.0/((*x)[0]*((*x)[0]-2.0) + 2.0);
    W_epetracrs->ReplaceMyValues(0, 1, &value, &index);
  }

}

#endif
