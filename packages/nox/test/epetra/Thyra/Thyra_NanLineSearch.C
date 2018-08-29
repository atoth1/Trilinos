//@HEADER
// ************************************************************************
//
//            NOX: An Object-Oriented Nonlinear Solver Package
//                 Copyright (2002) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// Questions? Contact Roger Pawlowski (rppawlo@sandia.gov) or
// Eric Phipps (etphipp@sandia.gov), Sandia National Laboratories.
// ************************************************************************
//  CVS Information
//  $Source$
//  $Author$
//  $Date$
//  $Revision$
// ************************************************************************
//@HEADER

// NOX Objects
#include "NOX.H"
#include "NOX_Thyra.H"

// Trilinos Objects
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif
#include "Stratimikos_DefaultLinearSolverBuilder.hpp"
#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include "Thyra_VectorStdOps.hpp"

#include "ModelEvaluatorBoundedArcTan.hpp"


TEUCHOS_UNIT_TEST(NOX_Thyra_BoundedArcTan, Backtrack)
{
  Teuchos::TimeMonitor::zeroOutTimers();

  // Create a communicator for Epetra objects
#ifdef HAVE_MPI
  Epetra_MpiComm Comm( MPI_COMM_WORLD );
#else
  Epetra_SerialComm Comm;
#endif

  // Check we have only one processor since this problem doesn't work
  // for more than one proc
  TEST_ASSERT(Comm.NumProc() == 1);

  // Create the model evaluator object
  Teuchos::RCP<ModelEvaluatorBoundedArcTan<double> > thyraModel =
    Teuchos::rcp(new ModelEvaluatorBoundedArcTan<double>(Teuchos::rcpFromRef(Comm)));

  Stratimikos::DefaultLinearSolverBuilder builder;

  Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::parameterList();
  p->set("Linear Solver Type", "AztecOO");
  p->set("Preconditioner Type", "None");
  builder.setParameterList(p);

  Teuchos::RCP< Thyra::LinearOpWithSolveFactoryBase<double> >
    lowsFactory = builder.createLinearSolveStrategy("");

  thyraModel->set_W_factory(lowsFactory);

  // Create the initial guess
  Teuchos::RCP<Thyra::VectorBase<double> >
    initial_guess = thyraModel->getNominalValues().get_x()->clone_v();

  // Create the NOX::Thyra::Group
  Teuchos::RCP<NOX::Thyra::Group> nox_group =
    Teuchos::rcp(new NOX::Thyra::Group(*initial_guess, thyraModel));

  // Create the NOX status tests and the solver
  // Create the convergence tests
  Teuchos::RCP<NOX::StatusTest::NormF> absresid =
    Teuchos::rcp(new NOX::StatusTest::NormF(1.0e-8));
  Teuchos::RCP<NOX::StatusTest::MaxIters> maxiters =
    Teuchos::rcp(new NOX::StatusTest::MaxIters(20));
  Teuchos::RCP<NOX::StatusTest::FiniteValue> fv =
    Teuchos::rcp(new NOX::StatusTest::FiniteValue);
  Teuchos::RCP<NOX::StatusTest::Combo> combo =
    Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR));
  combo->addStatusTest(fv);
  combo->addStatusTest(absresid);
  combo->addStatusTest(maxiters);

  // Create nox parameter list
  Teuchos::RCP<Teuchos::ParameterList> nl_params = Teuchos::parameterList();
  nl_params->set("Nonlinear Solver", "Line Search Based");
  nl_params->sublist("Line Search").set("Method", "Backtrack");
  nl_params->sublist("Printing").sublist("Output Information").set("Details",true);
  nl_params->sublist("Printing").sublist("Output Information").set("Outer Iteration",true);
  nl_params->sublist("Printing").sublist("Output Information").set("Inner Iteration",true);

  // Create the solver
  Teuchos::RCP<NOX::Solver::Generic> solver =
    NOX::Solver::buildSolver(nox_group, combo, nl_params);
  NOX::StatusTest::StatusType solvStatus = solver->solve();

  TEST_ASSERT(solvStatus == NOX::StatusTest::Converged);
  TEST_ASSERT(solver->getNumIterations() == 5);
  const double val= Thyra::get_ele(
    dynamic_cast<const NOX::Thyra::Vector&>(solver->getSolutionGroup().getX()).getThyraVector(), 0);
  const double ans = 1.0;
  const double tol = 1.0e-8;
  TEST_FLOATING_EQUALITY(val, ans, tol);

  Teuchos::TimeMonitor::summarize();
}

TEUCHOS_UNIT_TEST(NOX_Thyra_NanLineSearch, Poly_Quadratic)
{
  Teuchos::TimeMonitor::zeroOutTimers();

  // Create a communicator for Epetra objects
#ifdef HAVE_MPI
  Epetra_MpiComm Comm( MPI_COMM_WORLD );
#else
  Epetra_SerialComm Comm;
#endif

  // Check we have only one processor since this problem doesn't work
  // for more than one proc
  TEST_ASSERT(Comm.NumProc() == 1);

  // Create the model evaluator object
  Teuchos::RCP<ModelEvaluatorBoundedArcTan<double> > thyraModel =
    Teuchos::rcp(new ModelEvaluatorBoundedArcTan<double>(Teuchos::rcpFromRef(Comm)));

  Stratimikos::DefaultLinearSolverBuilder builder;

  Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::parameterList();
  p->set("Linear Solver Type", "AztecOO");
  p->set("Preconditioner Type", "None");
  builder.setParameterList(p);

  Teuchos::RCP< Thyra::LinearOpWithSolveFactoryBase<double> >
    lowsFactory = builder.createLinearSolveStrategy("");

  thyraModel->set_W_factory(lowsFactory);

  // Create the initial guess
  Teuchos::RCP<Thyra::VectorBase<double> >
    initial_guess = thyraModel->getNominalValues().get_x()->clone_v();

  // Create the NOX::Thyra::Group
  Teuchos::RCP<NOX::Thyra::Group> nox_group =
    Teuchos::rcp(new NOX::Thyra::Group(*initial_guess, thyraModel));

  // Create the NOX status tests and the solver
  // Create the convergence tests
  Teuchos::RCP<NOX::StatusTest::NormF> absresid =
    Teuchos::rcp(new NOX::StatusTest::NormF(1.0e-8));
  Teuchos::RCP<NOX::StatusTest::MaxIters> maxiters =
    Teuchos::rcp(new NOX::StatusTest::MaxIters(200));
  Teuchos::RCP<NOX::StatusTest::FiniteValue> fv =
    Teuchos::rcp(new NOX::StatusTest::FiniteValue);
  Teuchos::RCP<NOX::StatusTest::Combo> combo =
    Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR));
  combo->addStatusTest(fv);
  combo->addStatusTest(absresid);
  combo->addStatusTest(maxiters);

  // Create nox parameter list
  Teuchos::RCP<Teuchos::ParameterList> nl_params = Teuchos::parameterList();
  nl_params->set("Nonlinear Solver", "Line Search Based");
  nl_params->sublist("Line Search").set("Method", "Polynomial");
  nl_params->sublist("Line Search").sublist("Polynomial").set("Interpolation Type", "Quadratic");
  // Setting large reduction factor to force line search to actually perform interpolations
  nl_params->sublist("Line Search").sublist("Polynomial").set("Alpha Factor", 0.9);
  nl_params->sublist("Line Search").sublist("Polynomial").set("Max Iters", 10);
  nl_params->sublist("Printing").sublist("Output Information").set("Details",true);
  nl_params->sublist("Printing").sublist("Output Information").set("Outer Iteration",true);
  nl_params->sublist("Printing").sublist("Output Information").set("Inner Iteration",true);

  // Create the solver
  Teuchos::RCP<NOX::Solver::Generic> solver =
    NOX::Solver::buildSolver(nox_group, combo, nl_params);
  NOX::StatusTest::StatusType solvStatus = solver->solve();

  TEST_ASSERT(solvStatus == NOX::StatusTest::Converged);
  TEST_ASSERT(solver->getNumIterations() == 126);
  const double val= Thyra::get_ele(
    dynamic_cast<const NOX::Thyra::Vector&>(solver->getSolutionGroup().getX()).getThyraVector(), 0);
  const double ans = 1.0;
  const double tol = 1.0e-8;
  TEST_FLOATING_EQUALITY(val, ans, tol);

  Teuchos::TimeMonitor::summarize();
}

TEUCHOS_UNIT_TEST(NOX_Thyra_NanLineSearch, Poly_Quadratic3)
{
  Teuchos::TimeMonitor::zeroOutTimers();

  // Create a communicator for Epetra objects
#ifdef HAVE_MPI
  Epetra_MpiComm Comm( MPI_COMM_WORLD );
#else
  Epetra_SerialComm Comm;
#endif

  // Check we have only one processor since this problem doesn't work
  // for more than one proc
  TEST_ASSERT(Comm.NumProc() == 1);

  // Create the model evaluator object
  Teuchos::RCP<ModelEvaluatorBoundedArcTan<double> > thyraModel =
    Teuchos::rcp(new ModelEvaluatorBoundedArcTan<double>(Teuchos::rcpFromRef(Comm)));

  Stratimikos::DefaultLinearSolverBuilder builder;

  Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::parameterList();
  p->set("Linear Solver Type", "AztecOO");
  p->set("Preconditioner Type", "None");
  builder.setParameterList(p);

  Teuchos::RCP< Thyra::LinearOpWithSolveFactoryBase<double> >
    lowsFactory = builder.createLinearSolveStrategy("");

  thyraModel->set_W_factory(lowsFactory);

  // Create the initial guess
  Teuchos::RCP<Thyra::VectorBase<double> >
    initial_guess = thyraModel->getNominalValues().get_x()->clone_v();

  // Create the NOX::Thyra::Group
  Teuchos::RCP<NOX::Thyra::Group> nox_group =
    Teuchos::rcp(new NOX::Thyra::Group(*initial_guess, thyraModel));

  // Create the NOX status tests and the solver
  // Create the convergence tests
  Teuchos::RCP<NOX::StatusTest::NormF> absresid =
    Teuchos::rcp(new NOX::StatusTest::NormF(1.0e-8));
  Teuchos::RCP<NOX::StatusTest::MaxIters> maxiters =
    Teuchos::rcp(new NOX::StatusTest::MaxIters(200));
  Teuchos::RCP<NOX::StatusTest::FiniteValue> fv =
    Teuchos::rcp(new NOX::StatusTest::FiniteValue);
  Teuchos::RCP<NOX::StatusTest::Combo> combo =
    Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR));
  combo->addStatusTest(fv);
  combo->addStatusTest(absresid);
  combo->addStatusTest(maxiters);

  // Create nox parameter list
  Teuchos::RCP<Teuchos::ParameterList> nl_params = Teuchos::parameterList();
  nl_params->set("Nonlinear Solver", "Line Search Based");
  nl_params->sublist("Line Search").set("Method", "Polynomial");
  nl_params->sublist("Line Search").sublist("Polynomial").set("Interpolation Type", "Quadratic3");
  // Setting large reduction factor to force line search to actually perform interpolations
  nl_params->sublist("Line Search").sublist("Polynomial").set("Alpha Factor", 0.9);
  nl_params->sublist("Line Search").sublist("Polynomial").set("Max Iters", 10);
  nl_params->sublist("Printing").sublist("Output Information").set("Details",true);
  nl_params->sublist("Printing").sublist("Output Information").set("Outer Iteration",true);
  nl_params->sublist("Printing").sublist("Output Information").set("Inner Iteration",true);

  // Create the solver
  Teuchos::RCP<NOX::Solver::Generic> solver =
    NOX::Solver::buildSolver(nox_group, combo, nl_params);
  NOX::StatusTest::StatusType solvStatus = solver->solve();

  TEST_ASSERT(solvStatus == NOX::StatusTest::Converged);
  TEST_ASSERT(solver->getNumIterations() == 126);
  const double val= Thyra::get_ele(
    dynamic_cast<const NOX::Thyra::Vector&>(solver->getSolutionGroup().getX()).getThyraVector(), 0);
  const double ans = 1.0;
  const double tol = 1.0e-8;
  TEST_FLOATING_EQUALITY(val, ans, tol);

  Teuchos::TimeMonitor::summarize();
}

TEUCHOS_UNIT_TEST(NOX_Thyra_NanLineSearch, Poly_Cubic)
{
  Teuchos::TimeMonitor::zeroOutTimers();

  // Create a communicator for Epetra objects
#ifdef HAVE_MPI
  Epetra_MpiComm Comm( MPI_COMM_WORLD );
#else
  Epetra_SerialComm Comm;
#endif

  // Check we have only one processor since this problem doesn't work
  // for more than one proc
  TEST_ASSERT(Comm.NumProc() == 1);

  // Create the model evaluator object
  Teuchos::RCP<ModelEvaluatorBoundedArcTan<double> > thyraModel =
    Teuchos::rcp(new ModelEvaluatorBoundedArcTan<double>(Teuchos::rcpFromRef(Comm)));

  Stratimikos::DefaultLinearSolverBuilder builder;

  Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::parameterList();
  p->set("Linear Solver Type", "AztecOO");
  p->set("Preconditioner Type", "None");
  builder.setParameterList(p);

  Teuchos::RCP< Thyra::LinearOpWithSolveFactoryBase<double> >
    lowsFactory = builder.createLinearSolveStrategy("");

  thyraModel->set_W_factory(lowsFactory);

  // Create the initial guess
  Teuchos::RCP<Thyra::VectorBase<double> >
    initial_guess = thyraModel->getNominalValues().get_x()->clone_v();

  // Create the NOX::Thyra::Group
  Teuchos::RCP<NOX::Thyra::Group> nox_group =
    Teuchos::rcp(new NOX::Thyra::Group(*initial_guess, thyraModel));

  // Create the NOX status tests and the solver
  // Create the convergence tests
  Teuchos::RCP<NOX::StatusTest::NormF> absresid =
    Teuchos::rcp(new NOX::StatusTest::NormF(1.0e-8));
  Teuchos::RCP<NOX::StatusTest::MaxIters> maxiters =
    Teuchos::rcp(new NOX::StatusTest::MaxIters(200));
  Teuchos::RCP<NOX::StatusTest::FiniteValue> fv =
    Teuchos::rcp(new NOX::StatusTest::FiniteValue);
  Teuchos::RCP<NOX::StatusTest::Combo> combo =
    Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR));
  combo->addStatusTest(fv);
  combo->addStatusTest(absresid);
  combo->addStatusTest(maxiters);

  // Create nox parameter list
  Teuchos::RCP<Teuchos::ParameterList> nl_params = Teuchos::parameterList();
  nl_params->set("Nonlinear Solver", "Line Search Based");
  nl_params->sublist("Line Search").set("Method", "Polynomial");
  nl_params->sublist("Line Search").sublist("Polynomial").set("Interpolation Type", "Cubic");
  // Setting large reduction factor to force line search to actually perform interpolations
  nl_params->sublist("Line Search").sublist("Polynomial").set("Alpha Factor", 0.9);
  nl_params->sublist("Line Search").sublist("Polynomial").set("Max Iters", 10);
  nl_params->sublist("Printing").sublist("Output Information").set("Details",true);
  nl_params->sublist("Printing").sublist("Output Information").set("Outer Iteration",true);
  nl_params->sublist("Printing").sublist("Output Information").set("Inner Iteration",true);

  // Create the solver
  Teuchos::RCP<NOX::Solver::Generic> solver =
    NOX::Solver::buildSolver(nox_group, combo, nl_params);
  NOX::StatusTest::StatusType solvStatus = solver->solve();

  TEST_ASSERT(solvStatus == NOX::StatusTest::Converged);
  TEST_ASSERT(solver->getNumIterations() == 126);
  const double val= Thyra::get_ele(
    dynamic_cast<const NOX::Thyra::Vector&>(solver->getSolutionGroup().getX()).getThyraVector(), 0);
  const double ans = 1.0;
  const double tol = 1.0e-8;
  TEST_FLOATING_EQUALITY(val, ans, tol);

  Teuchos::TimeMonitor::summarize();
}
