from dataclasses import dataclass
from functools import cached_property, partial
from typing import Callable, Final, Literal, Tuple, Type, Union, overload

import numpy as np
from scipy.linalg import get_lapack_funcs
from scipy.sparse import csr_matrix

from pylops import LinearOperator
from pylops.utils.backend import get_normalize_axis_index
from pylops.utils.decorators import reshaped
from pylops.utils.typing import Float64Vector, Int64Vector

ONE_SIXTH: Final[float] = 1.0 / 6.0
TWO_THIRDS: Final[float] = 2.0 / 3.0


_InexactVector = np.ndarray[tuple[int], np.dtype[np.inexact]]
_InexactMatrix = np.ndarray[tuple[int, int], np.dtype[np.inexact]]
_InexactArray = Union[_InexactVector, _InexactMatrix]


@overload
def _second_order_finite_differences_zero_padded(
    x: _InexactVector,
    pad_width: tuple[tuple[int, int], ...],
) -> _InexactVector:
    ...


@overload
def _second_order_finite_differences_zero_padded(
    x: _InexactMatrix,
    pad_width: tuple[tuple[int, int], ...],
) -> _InexactMatrix:
    ...


def _second_order_finite_differences_zero_padded(
    x: _InexactArray,
    pad_width: tuple[tuple[int, int], ...],
) -> _InexactArray:
    """
    Computes the second order finite differences of ``x`` along axis 0 and pads the
    result with ``pad_width[0][0]`` leading and `pad_width[0][1]`` trailing zeros
    along the same axis.

    Parameters
    ----------
    x : :obj:`numpy.ndarray` of shape ``(n,)`` or shape ``(m, n)``
        The input array.
        It is processed along axis 0.
    pad_width : ((:obj:`int`, :obj:`int`), ...) of length ``x.ndim``
        The ``pad_width`` argument to pass to ``numpy.pad`` to achieve the subsequent
        zero padding along axis 0.

    Returns
    -------
    x_diffs : :obj:`numpy.ndarray` of shape ``(n,)`` or shape ``(m, n)``
        The second order finite differences of ``x`` padded with leading and trailing
        zeros along axis 0.

    """

    return np.pad(
        np.diff(
            x,
            n=2,
            axis=0,
        ),
        pad_width=pad_width,
        mode="constant",
        constant_values=0.0,
    )


@overload
def _second_order_finite_differences_zero_padded_transposed(
    x: _InexactVector,
    x_slice: slice,
    pad_width: tuple[tuple[int, int], ...],
) -> _InexactVector:
    ...


@overload
def _second_order_finite_differences_zero_padded_transposed(
    x: _InexactMatrix,
    x_slice: slice,
    pad_width: tuple[tuple[int, int], ...],
) -> _InexactMatrix:
    ...


def _second_order_finite_differences_zero_padded_transposed(
    x: _InexactArray,
    x_slice: slice,
    pad_width: tuple[tuple[int, int], ...],
) -> _InexactArray:
    """
    Computes the transposed operation of the second order finite differences operator
    with subsequent zero padding along axis 0, i.e.,

    - ``x[0 : x_slices[axis].start,]`` and ``x[x_slices[axis].stop ::]`` are not
        considered,
    - ``x[x_slice]`` is padded with ``pad_width[0][0]`` leading and ``pad_width[0][1]``
        trailing zeros,
    - the second order finite differences of the padded array are computed

    Parameters
    ----------
    x : :obj:`numpy.ndarray` of shape ``(n,)``  or shape ``(m, n)``
        The input array.
        It is processed along axis 0.
    x_slices : (:obj:`slice`, ...) of length ``x.ndim``
        The slices to extract from ``x`` along each dimension to ignore leading and
        trailing elements along axis 0.
    pad_width : ((:obj:`int`, :obj:`int`), ...) of length ``x.ndim``
        The ``pad_width`` argument to pass to ``numpy.pad`` to achieve the zero
        padding along axis 0.

    Returns
    -------
    x_diffs : :obj:`numpy.ndarray` of shape ``(n,)``  or shape ``(m, n)``
        The result of the transposed second order finite differences operator with
        subsequent zero padding along axis 0.

    """

    return np.diff(
        np.pad(
            x[x_slice],
            pad_width=pad_width,
            mode="constant",
            constant_values=0.0,
        ),
        n=2,
        axis=0,
    )


@dataclass
class _TridiagonalMatrix:
    """
    Represents a tridiagonal matrix with

    - the main diagonal ``.main_diagonal``,
    - the super-diagonal ``.super_diagonal``,
    - the sub-diagonal ``.sub_diagonal``.

    """

    main_diagonal: _InexactVector
    super_diagonal: _InexactVector
    sub_diagonal: _InexactVector

    def __post_init__(self) -> None:
        """
        Validates the input.

        """

        for which in ("main", "super", "sub"):
            ndim = getattr(self, f"{which}_diagonal").ndim
            if ndim != 1:
                raise ValueError(
                    f"Expected {which} diagonal to be a 1-dimensional Array, but it is "
                    f"{ndim}-dimensional."
                )

        main_diagonal_size = self.main_diagonal.size
        for which in ("super", "sub"):
            size = getattr(self, f"{which}_diagonal").size
            if size != main_diagonal_size - 1:
                raise ValueError(
                    f"Expected {which} diagonal to have 1 entry less than the main "
                    f"diagonal, but it has {size} entries and the main diagonal has "
                    f"{main_diagonal_size} entries."
                )

        return

    def __iter__(self):
        """
        Returns an iterator over the sub-diagonal, main diagonal and super-diagonal
        (in that order) as required for the LAPACK routines ``?gttrf``.

        """

        yield self.sub_diagonal
        yield self.main_diagonal
        yield self.super_diagonal

        return

    @property
    def T(self) -> "_TridiagonalMatrix":
        """
        Returns the transpose of the tridiagonal matrix.

        """

        return _TridiagonalMatrix(
            main_diagonal=self.main_diagonal,
            super_diagonal=self.sub_diagonal,
            sub_diagonal=self.super_diagonal,
        )


@dataclass
class _TridiagonalLUDecomposition:
    """
    Represents the LU decomposition of a tridiagonal matrix as performed by the LAPACK
    routines ``?gttrf``.

    """

    sub_diagonal_lu: _InexactVector
    main_diagonal_lu: _InexactVector
    super_diagonal_lu: _InexactVector
    super_two_diagonal_lu: _InexactVector
    pivot_indices: np.ndarray[tuple[int], np.dtype[np.int64]]

    def __iter__(self):
        """
        Returns an iterator over the sub-diagonal, main diagonal, super-diagonal,
        the second super-diagonal (filled by pivoting) and the pivot indices
        (in that order) as required for the LAPACK routines ``?gttrs``.

        """

        yield self.sub_diagonal_lu
        yield self.main_diagonal_lu
        yield self.super_diagonal_lu
        yield self.super_two_diagonal_lu
        yield self.pivot_indices

        return

    @staticmethod
    def from_tridiagonal_matrix(
        matrix: _TridiagonalMatrix,
        lapack_factorizer: Callable,
    ) -> "_TridiagonalLUDecomposition":
        """
        Computes the LU decomposition of a tridiagonal matrix using the LAPACK routine
        ``?gttrf``.

        Parameters
        ----------
        matrix : :obj:`TridiagonalMatrix`
            The tridiagonal matrix to decompose.
        lapack_factorizer : callable
            The LAPACK routine ``?gttrf`` to use for the decomposition.

        Returns
        -------
        lu_decomposition : :obj:`TridiagonalLUDecomposition`
            The LU decomposition of the tridiagonal matrix.

        """

        (
            sub_diagonal_lu,
            main_diagonal_lu,
            super_diagonal_lu,
            super_two_diagonal_lu,
            pivot_indices,
            info,
        ) = lapack_factorizer(*matrix)

        if info == 0:
            return _TridiagonalLUDecomposition(
                sub_diagonal_lu=sub_diagonal_lu,
                main_diagonal_lu=main_diagonal_lu,
                super_diagonal_lu=super_diagonal_lu,
                super_two_diagonal_lu=super_two_diagonal_lu,
                pivot_indices=pivot_indices,
            )

        raise np.linalg.LinAlgError(
            f"Could not LU-factorize tridiagonal matrix! Got {info = }."
        )

    @overload
    def solve(
        self,
        rhs: _InexactVector,
        lapack_solver: Callable,
    ) -> _InexactVector:
        ...

    @overload
    def solve(
        self,
        rhs: _InexactMatrix,
        lapack_solver: Callable,
    ) -> _InexactMatrix:
        ...

    def solve(
        self,
        rhs: _InexactArray,
        lapack_solver: Callable,
    ) -> _InexactArray:
        """
        Solves the linear system of equations ``A @ x = rhs`` where ``A`` is the
        tridiagonal matrix represented by the LU decomposition. For this, the LAPACK
        routine ``?gttrs`` is used.

        Parameters
        ----------
        rhs : :obj:`numpy.ndarray` of shape ``(n,)``  or shape ``(m, n)``
            The right-hand side(s) of the linear system of equations.
            It is processed along axis 0, i.e.,

            - a 1D-Array is treated as a single right hand side.
            - each colum of a 2D-Array is treated as a single right hand side.

        lapack_solver : callable
            The LAPACK routine ``?gttrs`` to use for solving the system.

        Returns
        -------
        x : :obj:`numpy.ndarray` of shape ``(n,)``  or shape ``(m, n)``
            The solution of the linear system(s) of equations.
            For 2D-Arrays, the ``j``-th column is the solution of the respective
            ``rhs[::, j]``.

        """

        x, info = lapack_solver(*self, rhs)

        if info == 0:
            return x

        raise np.linalg.LinAlgError(
            f"Could not solve LU-factorization of tridiagonal matrix! Got {info = }."
        )


def _make_cubic_spline_left_hand_side(
    dims: int,
) -> _TridiagonalMatrix:
    """
    Constructs the banded matrix ``A`` for the linear system of equations ``A @ m = b``
    where

    - ``A`` is a diagonally dominant tridiagonal matrix with the main diagonal
        being ``[1, 2/3, 2/3, ..., 2/3, 2/3, 1]``, the super-diagonal being
        ``[0, 1/6, 1/6, ..., 1/6, 1/6]`` and the sub-diagonal being
        ``[1/6, 1/6, ..., 1/6, 1/6, 0]``,
    - ``m`` is the unknown vector of spline coefficients,
    - ``b`` is the second order finite differences of the ``y`` values for which the
        spline should interpolate.

    Parameters
    ----------
    dims : :obj:`int`
        The number of points the spline should interpolate.

    Returns
    -------
    matrix_a : :obj:`TridiagonalMatrix`
        The tridiagonal matrix ``A``.

    """

    main_diag = np.empty(shape=(dims,), dtype=np.float64)
    super_diag = np.empty(shape=(dims - 1,), dtype=np.float64)

    main_diag[0] = 1.0
    main_diag[1 : dims - 1] = TWO_THIRDS
    main_diag[dims - 1] = 1.0

    super_diag[0] = 0.0
    super_diag[1 : dims - 1] = ONE_SIXTH

    return _TridiagonalMatrix(
        main_diagonal=main_diag,
        super_diagonal=super_diag,
        sub_diagonal=np.flip(super_diag).copy(),  # to avoid a view
    )


def _make_cubic_spline_x_csr(
    dims: int,
    iava: Float64Vector,
    base_indices: Int64Vector,
    iava_remainders: Float64Vector,
) -> csr_matrix:
    """
    Constructs the specifications ``data``, ``indices``, and ``indptr`` for a
    :obj:`scipy.sparse.csr_matrix` ``X`` that can be used to interpolate the
    equally spaced ``y`` values with a cubic spline like ``X @ Q @ y`` where

    - ``X`` is the interpolation matrix to be constructed,
    - ``Q`` is the linear operator that obtains the spline coefficients ``y``
        (a vertical concatenation of the ``y`` values and the spline coefficients
        ``m``),
    - ``y`` are the values to be interpolated.

    Parameters
    ----------
    dims : :obj:`int`
        The number of points the spline interpolation takes as input.
    iava : :obj:`numpy.ndarray` of shape ``(n,)`` and dtype ``numpy.float64``
        Floating indices of the locations to which the spline should interpolate.
    base_indices : :obj:`numpy.ndarray` of shape ``(n,)`` and dtype ``numpy.int64``
        The indices of the respective first data point in ``y`` for the intervals
        in which the corresponding ``iava`` values lie.
    iava_remainders : :obj:`numpy.ndarray` of shape ``(n,)`` and dtype ``numpy.float64``
        The remainders of the ``iava`` values after subtracting the respective
        ``base_indices``.

    Returns
    -------
    cubic_spline_interp_matrix : :obj:`scipy.sparse.csr_matrix`
        The ``X`` is CSR-format.

    """

    # some auxiliary variables are required
    iava_remainders_cubic = (  # (x - x[i])³
        iava_remainders * iava_remainders * iava_remainders
    )
    one_minus_iava_remainders = 1.0 - iava_remainders  # (x[i + 1] - x)
    one_minus_iava_remainders_cubic = (  # (x[i + 1] - x)³
        one_minus_iava_remainders
        * one_minus_iava_remainders
        * one_minus_iava_remainders
    )

    # for each data point, except for the first and the last one, we need 4 entries
    # to multiply with ``y[i]``, ``y[i + 1]``, ``m[i]``, and ``m[i + 1]``;

    data = np.column_stack(
        (
            one_minus_iava_remainders,
            iava_remainders,
            ONE_SIXTH * (one_minus_iava_remainders_cubic - one_minus_iava_remainders),
            ONE_SIXTH * (iava_remainders_cubic - iava_remainders),
        )
    ).ravel()

    indices = np.add.outer(
        base_indices,
        np.array([0, 1, dims, dims + 1], dtype=np.int64),
    ).ravel()

    indptr = np.arange(0, 4 * (iava.size + 1), 4, dtype=np.int64)

    return csr_matrix(
        (
            data,
            indices,
            indptr,
        ),
        shape=(iava.size, 2 * dims),
    )


class CubicSplineInterpolator(LinearOperator):
    """
    Custom cubic spline interpolator.

    Parameters
    ----------
    dims : :obj:`int`
        The number of points the spline should interpolate.
    iava : Array-like of shape ``(n,)``
        Floating indices of the locations to which the spline should interpolate.
    dtype :
        The data type of the input and output arrays.

    """

    def __init__(
        self,
        dims: Tuple,
        dimsd: Tuple,
        iava: Float64Vector,
        axis: Literal[0, 1],
        dtype: Type,
        name: str,
    ) -> None:

        # --- Operator Initialization ---

        self.dims: Tuple = dims
        self.dimsd: Tuple = dimsd
        self.iava: Float64Vector = iava
        self.axis: int = get_normalize_axis_index()(axis, len(dims))

        ndim = len(self.dims)
        num_cols = self.dims[self.axis]

        super().__init__(
            dtype=dtype,
            dims=dims,
            dimsd=dimsd,
            name=name,
        )

        # --- Pre-computations of the Tridiagonal Systems ---

        # NOTE: the LU-factorization will always be performed on ``float64`` while
        #       the LU-solve type depends on the actual dtype, which might also be
        #       complex
        self._tridiag_factorize = get_lapack_funcs(("gttrf",), dtype=np.float64)[0]
        self._tridiag_lu_solve = get_lapack_funcs(("gttrs",), dtype=self.dtype)[0]

        lhs_matrix: _TridiagonalMatrix = _make_cubic_spline_left_hand_side(
            dims=num_cols
        )
        self.lhs_matrix_lu = _TridiagonalLUDecomposition.from_tridiagonal_matrix(
            matrix=lhs_matrix,
            lapack_factorizer=self._tridiag_factorize,
        )
        self.lhs_matrix_transposed_lu = (
            _TridiagonalLUDecomposition.from_tridiagonal_matrix(
                matrix=lhs_matrix.T,
                lapack_factorizer=self._tridiag_factorize,
            )
        )

        # --- Pre-computation of the Interpolator Matrices ---

        base_indices = np.clip(
            self.iava.astype(np.int64),  # already rounds down
            a_min=0,
            a_max=num_cols - 2,
        )

        self.X_matrix: csr_matrix = _make_cubic_spline_x_csr(
            dims=num_cols,
            iava=self.iava,
            base_indices=base_indices,
            iava_remainders=self.iava - base_indices,
        )
        self.X_matrix_transposed: csr_matrix = self.X_matrix.transpose().tocsr()  # type: ignore

        self.matvec_difference_method = partial(
            _second_order_finite_differences_zero_padded,
            pad_width=((1, 1),),
        )
        self.rmatvec_difference_method = partial(
            _second_order_finite_differences_zero_padded_transposed,
            x_slice=slice(1, num_cols - 1),
            pad_width=((2, 2),),
        )

        self.matmat_difference_method = partial(
            _second_order_finite_differences_zero_padded,
            pad_width=tuple([(1, 1)] + [(0, 0) for _ in range(0, ndim - 1)]),
        )
        self.rmatmat_difference_method = partial(
            _second_order_finite_differences_zero_padded_transposed,
            x_slice=slice(1, num_cols - 1),
            pad_width=tuple([(2, 2)] + [(0, 0) for _ in range(0, ndim - 1)]),
        )

    @cached_property
    def num_cols(self) -> int:
        return self.dims[self.axis]

    @reshaped(swapaxis=True, axis=0)
    def _matvec(self, x: _InexactArray) -> _InexactArray:
        m_coeffs = self.lhs_matrix_lu.solve(
            rhs=self.matmat_difference_method(x).reshape(x.shape[0], -1),
            lapack_solver=self._tridiag_lu_solve,
        )
        return (
            self.X_matrix
            @ np.concatenate(
                (x.reshape(x.shape[0], -1), m_coeffs),
                axis=0,
            )
        ).reshape(-1, *x.shape[1:])

    def _matmat(self, x: _InexactArray) -> _InexactArray:
        m_coeffs = self.lhs_matrix_lu.solve(
            rhs=self.matmat_difference_method(x),
            lapack_solver=self._tridiag_lu_solve,
        )

        return self.X_matrix @ np.concatenate(
            (x, m_coeffs),
            axis=0,
        )

    @reshaped(swapaxis=True, axis=0)
    def _rmatvec(self, x: _InexactArray) -> _InexactArray:
        shape = (self.num_cols, *x.shape[1:])
        x_mod = self.X_matrix_transposed @ x.reshape(x.shape[0], -1)
        return x_mod[0 : self.num_cols].reshape(shape) + self.rmatmat_difference_method(
            self.lhs_matrix_transposed_lu.solve(
                rhs=x_mod[self.num_cols : x_mod.size],
                lapack_solver=self._tridiag_lu_solve,
            ).reshape(shape)
        )

    def _rmatmat(self, x: _InexactArray) -> _InexactArray:
        x_mod = self.X_matrix_transposed @ x
        return x_mod[0 : self.num_cols] + self.rmatmat_difference_method(
            self.lhs_matrix_transposed_lu.solve(
                rhs=x_mod[self.num_cols : x_mod.size],
                lapack_solver=self._tridiag_lu_solve,
            )
        )
