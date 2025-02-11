// ==================================
// Creating and tabulating an element
// ==================================
//
// This demo shows how Basix can be used to create an element
// and tabulate the values of its basis functions at a set of
// points.

#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <basix/mdspan.hpp>
#include <iostream>

#include "poisson.h"

namespace stdex
    = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE;
template <typename T, std::size_t d>
using mdspan_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, d>>;

using T = double;

T compute_detJ(std::span<const T> coordinate_dofs)
{
  auto p0 = coordinate_dofs.subspan(0, 3);
  auto p1 = coordinate_dofs.subspan(3, 3);
  auto p2 = coordinate_dofs.subspan(6, 3);

  T _detJ = (p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1]);

  return _detJ;
}

int main(int argc, char* argv[])
{
  // Create a degree 4 Lagrange element on a quadrilateral
  // For Lagrange elements, we use `basix::element::family::P`.
  auto family = basix::element::family::P;
  auto cell_type = basix::cell::type::triangle;
  int k = 1;

  // For Lagrange elements, we must provide and extra argument: the Lagrange
  // variant. In this example, we use the equispaced variant: this will place
  // the degrees of freedom (DOFs) of the element in an equally spaced lattice.
  auto variant = basix::element::lagrange_variant::unset;

  // Create the lagrange element
  basix::FiniteElement lagrange = basix::create_element<T>(
      family, cell_type, k, variant, basix::element::dpc_variant::unset, false);

  // Get the number of degrees of freedom for the element
  int dofs = lagrange.dim();

  // Create a set of points, and tabulate the basis functions
  // of the Lagrange element at these points.
  const auto [points, wts] = basix::quadrature::make_quadrature<T>(
      basix::quadrature::type::Default, cell_type, basix::polyset::type::standard, 1);

  int num_points = wts.size();

  const ufcx_form& ufcx_a = *form_poisson_a;
  const int* integral_offsets = ufcx_a.form_integral_offsets;

  // ----------------------------------------------------------------------------------
  // Custom Integral
  // ----------------------------------------------------------------------------------
  ufcx_integral* custom_integral = ufcx_a.form_integrals[integral_offsets[cutcell]];
  ufcx_integral* cell_integral = ufcx_a.form_integrals[integral_offsets[cell]];

  std::cout << "fe_hash (generated)=" << custom_integral->finite_element_hashes[0] << std::endl;
  std::cout << "fe_hash (source)=" << lagrange.hash() << std::endl;

  auto [tab_data, shape] = lagrange.tabulate(custom_integral->finite_element_deriv_order[0], points, {points.size() / 2, 2});

  std::cout << "Tabulate data shape: [ ";
  for (auto s : shape)
    std::cout << s << " ";
  std::cout << "]" << std::endl;

  mdspan_t<const T, 4> tab(tab_data.data(), shape);
  std::cout << "Tabulate data (0, 0, :, 0): [ ";
  for (std::size_t i = 0; i < tab.extent(2); ++i)
    std::cout << tab(0, 0, i, 0) << " ";
  std::cout << "]" << std::endl;

  std::vector<T> coordinate_dofs = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  T _detJ = compute_detJ(coordinate_dofs);
  std::vector<T> weights(num_points);

  for(std::int32_t i = 0; i < num_points; ++i)
      weights[i] = wts[i]*std::fabs(_detJ);

  auto custom_kernel = custom_integral->tabulate_tensor_runtime_float64;
  auto cell_kernel = cell_integral->tabulate_tensor_float64;

  int num_dofs = shape[2];
  std::vector<T> A(num_dofs*num_dofs,0.0);

  custom_kernel(A.data(), {}, {}, coordinate_dofs.data(), nullptr,
        nullptr, &num_points,  points.data(), weights.data(), tab_data.data(), shape.data());

  std::vector<T> A_cell(num_dofs*num_dofs,0.0);
  cell_kernel(A_cell.data(), {}, {}, coordinate_dofs.data(), nullptr,
        nullptr);

  std::cout << "A=";
  for (std::size_t i = 0; i < A.size(); ++i)
    std::cout << A[i] << ", ";
  std::cout << std::endl;

  std::cout << "A_cell=";
  for (std::size_t i = 0; i < A.size(); ++i)
    std::cout << A_cell[i] << ", ";
  std::cout << std::endl;

  T sum_error = 0.0;

  for (std::size_t i = 0; i < A.size(); ++i)
     sum_error+=std::fabs(A[i]-A_cell[i]);

  std::cout << "Error sum=" << sum_error << std::endl;

  return 0;
}
