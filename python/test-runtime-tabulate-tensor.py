import sys

import basix
import basix.ufl
import cffi
import numpy as np
import ufl

import ffcx.codegeneration.jit
from ffcx.codegeneration.utils import dtype_to_c_type, dtype_to_scalar_dtype

def generate_kernel(forms, scalar_type, options):
    """Generate kernel for given forms."""
    compiled_forms, module, code = ffcx.codegeneration.jit.compile_forms(
        forms, options={"scalar_type": scalar_type}
    )

    for f, compiled_f in zip(forms, compiled_forms):
        assert compiled_f.rank == len(f.arguments())

    form0 = compiled_forms[0]

    offsets = form0.form_integral_offsets
    cell = module.lib.cell
    assert offsets[cell + 1] - offsets[cell] == 1
    integral_id = form0.form_integral_ids[offsets[cell]]
    assert integral_id == -1
    default_integral = form0.form_integrals[offsets[cell]]
    kernel = getattr(default_integral, f"tabulate_tensor_{scalar_type}")
    return kernel, code, module

def generate_runtime_kernel(forms, scalar_type, options):
    """Generate kernel for given forms."""
    compiled_forms, module, code = ffcx.codegeneration.jit.compile_forms(
        forms, options={"scalar_type": scalar_type}
    )

    for f, compiled_f in zip(forms, compiled_forms):
        assert compiled_f.rank == len(f.arguments())

    form0 = compiled_forms[0]

    offsets = form0.form_integral_offsets
    cutcell = module.lib.cutcell
    assert offsets[cutcell + 1] - offsets[cutcell] == 1
    integral_id = form0.form_integral_ids[offsets[cutcell]]
    assert integral_id == -1
    default_integral = form0.form_integrals[offsets[cutcell]]
    kernel = getattr(default_integral, f"tabulate_tensor_runtime_quad_{scalar_type}")
    return kernel, code, module, default_integral

# Create a simple form
mesh = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
e = basix.ufl.element("Lagrange", "triangle", 1)

V = ufl.FunctionSpace(mesh, e)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * (ufl.dx+ufl.dC)

# Generate and compile the kernel
dtype = "float64"
kernel, code, module = generate_kernel([a], dtype, {})

runtime_kernel, rt_code, rt_module, rt_integral = generate_runtime_kernel([a], dtype, {})

ffi = module.ffi
xdtype = dtype_to_scalar_dtype(dtype)
coords = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=xdtype)

c_type, c_xtype = dtype_to_c_type(dtype), dtype_to_c_type(xdtype)

A = np.zeros((3, 3), dtype=np.float64)
w = np.array([1.0, 1.0, 1.0], dtype=np.float64)
c = np.array([], dtype=np.float64)

print(c_type)

kernel(
    ffi.cast(f"{c_type} *", A.ctypes.data),
    ffi.cast(f"{c_type} *", w.ctypes.data),
    ffi.cast(f"{c_type} *", c.ctypes.data),
    ffi.cast(f"{c_xtype} *", coords.ctypes.data),
    ffi.NULL,
    ffi.NULL,
)

print(A)

A_rt = np.zeros((3, 3), dtype=np.float64)

points, weights = basix.make_quadrature(basix.CellType.triangle, 1)

print(points)
print(weights)

for i in range(0,rt_integral.num_fe):
  # find right element
  if(e.basix_hash()==rt_integral.finite_element_hashes[i]):
    print(rt_integral.finite_element_deriv_order[i])
    tbl = e._element.tabulate(rt_integral.finite_element_deriv_order[i],points)
    flat_tbl = tbl.flatten()
    shape = np.asarray(tbl.shape)

num_points = np.array([len(points)], dtype=np.intc)
shape = np.asarray(tbl.shape, dtype=np.uintp)

runtime_kernel(
    ffi.cast(f"{c_type} *", A_rt.ctypes.data),
    ffi.cast(f"{c_type} *", w.ctypes.data),
    ffi.cast(f"{c_type} *", c.ctypes.data),
    ffi.cast(f"{c_xtype} *", coords.ctypes.data),
    ffi.NULL,
    ffi.NULL,
    ffi.cast(f"int *", num_points.ctypes.data),
    ffi.cast(f"{c_type} *", points.ctypes.data),
    ffi.cast(f"{c_type} *", weights.ctypes.data),
    ffi.cast(f"{c_type} *", tbl.flatten().ctypes.data),
    ffi.cast(f"size_t *", np.asarray(tbl.shape).ctypes.data),
)

print(A_rt)