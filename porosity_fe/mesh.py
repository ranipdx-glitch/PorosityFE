"""Structured hex mesh and quality checks."""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ._ply_angles import _resolve_ply_angles
from .materials import MaterialProperties
from .porosity_field import PorosityField

logger = logging.getLogger("porosity_fe_analysis")

# ============================================================
# SECTION 4: MESH GENERATION
# ============================================================

class CompositeMesh:
    """3D structured hexahedral mesh of a composite coupon with porosity.

    Builds a regular grid of 8-node hexahedral elements over a
    rectangular coupon of dimensions ``L_x x L_y x L_z``, samples the
    porosity field at every node, assigns a ply id (and optional ply
    angle in degrees) to each element from its centroid z, and flags
    elements whose centroid falls inside any explicit
    :class:`VoidGeometry` for the explicit-inclusion solver path.

    The in-plane coupon size is **hard-coded** to ``L_x = 50.0`` mm and
    ``L_y = 20.0`` mm (a standard ASTM-style coupon). Through-thickness
    ``L_z`` is taken from ``material.total_thickness``
    (``t_ply * n_plies``). To analyze a different coupon size, set
    ``self.L_x`` / ``self.L_y`` on the instance and call
    :meth:`generate_mesh` again.

    Parameters
    ----------
    porosity_field : PorosityField
        Source of nodal porosity values. Sampled by
        :meth:`generate_mesh` at every node coordinate.
    material : MaterialProperties
        Composite material; supplies ``total_thickness`` (``L_z``) and
        ``n_plies`` for the per-element ply id assignment.
    nx, ny, nz : int, optional
        Number of elements along each axis (defaults
        ``nx=50``, ``ny=20``, ``nz=24``). Each must be a positive
        integer not greater than ``_MAX_ELEMENTS_PER_AXIS`` (10 000).
    ply_angles : list of float or {'QI', 'UD'}, optional
        Per-ply orientation in degrees, OR a string sentinel — ``'QI'``
        (default, expands to the 8-ply quasi-isotropic baseline
        ``[0, 90, 45, -45]_s``) or ``'UD'`` (all-zero unidirectional).
        Explicit lists shorter than ``n_plies`` are tiled. Passing
        ``None`` is deprecated and is currently resolved to all-zero
        plies (the historical default) with a
        :class:`DeprecationWarning` (#44 item 2); this back-compat path
        will be removed in a future major version.

    Attributes
    ----------
    porosity_field : PorosityField
        Stored input.
    material : MaterialProperties
        Stored input.
    nx, ny, nz : int
        Element counts along each axis.
    L_x, L_y, L_z : float
        Coupon dimensions in mm. ``L_x = 50.0`` and ``L_y = 20.0`` are
        defaults; ``L_z = material.total_thickness``.
    nodes : np.ndarray
        Shape ``(n_nodes, 3)`` float array of node coordinates (mm),
        populated by :meth:`generate_mesh`.
    elements : np.ndarray
        Shape ``(n_elem, 8)`` int array of node indices per element
        (VTK hexahedron ordering).
    porosity : np.ndarray
        Shape ``(n_nodes,)`` nodal porosity values in ``[0, 1]``.
    stiffness_reduction : np.ndarray
        Shape ``(n_nodes,)`` complementary ``1 - Vp`` field.
    ply_ids : np.ndarray
        Shape ``(n_nodes,)`` per-node ply id (``0`` to ``n_plies - 1``).
    elem_ply_ids : np.ndarray
        Shape ``(n_elem,)`` per-element ply id from the centroid z.
    ply_angles : np.ndarray
        Shape ``(n_elem,)`` per-element ply orientation in degrees.
    void_elements : np.ndarray
        Int indices of elements whose centroid lies inside any discrete
        void.
    void_element_set : set of int
        Same content as ``void_elements`` for O(1) membership tests.
    n_nodes, n_elements, n_dof : int
        Read-only sizes.

    Examples
    --------
    A default-size coupon with the T800 ply and 2 % uniform porosity:

    >>> mat = MATERIALS['T800_epoxy']
    >>> field = PorosityField(mat, void_volume_fraction=0.02,
    ...                       distribution='uniform')
    >>> mesh = CompositeMesh(field, mat, nx=10, ny=4, nz=4)
    >>> mesh.L_x, mesh.L_y
    (50.0, 20.0)
    >>> mesh.n_elements
    160

    Override the in-plane coupon size to 80 mm x 25 mm:

    >>> mesh = CompositeMesh(field, mat, nx=8, ny=4, nz=4)
    >>> mesh.L_x = 80.0
    >>> mesh.L_y = 25.0
    >>> mesh.generate_mesh()

    Notes
    -----
    ``ply_angles`` defaults — ``'QI'`` is the standardised default across
    :class:`EmpiricalSolver`, :class:`CompositeMesh`, and :class:`FESolver`
    (#44 item 2). The string sentinels expand to canonical baselines
    (``'QI'`` -> ``[0, 90, 45, -45]_s``; ``'UD'`` -> all-zero plies);
    explicit lists pass through unchanged. Pass ``ply_angles='UD'`` to
    reproduce the pre-#44 behaviour of leaving every element at
    ``ply_angle = 0``.
    """

    # Cap mesh dimensions to prevent accidental memory blowup. A million-element
    # mesh is already ~100x what the GUI spinboxes allow; an order of magnitude
    # above that is almost certainly a typo or unit confusion.
    _MAX_ELEMENTS_PER_AXIS = 10_000

    def __init__(self, porosity_field: PorosityField, material: MaterialProperties,
                 nx: int = 50, ny: int = 20, nz: int = 24,
                 ply_angles: Optional[Union[List[float], str]] = 'QI'):
        for axis_name, value in (('nx', nx), ('ny', ny), ('nz', nz)):
            if not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(
                    f"CompositeMesh.{axis_name} must be a positive integer "
                    f"(elements per axis), got {value!r}."
                )
            if value > self._MAX_ELEMENTS_PER_AXIS:
                raise ValueError(
                    f"CompositeMesh.{axis_name}={value} exceeds the "
                    f"{self._MAX_ELEMENTS_PER_AXIS} per-axis cap. "
                    f"Such a fine mesh would exhaust memory; "
                    f"reduce or split the analysis."
                )

        self.porosity_field = porosity_field
        self.material = material
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.L_x = 50.0
        self.L_y = 20.0
        self.L_z = material.total_thickness

        self.nodes = None
        self.elements = None
        self.porosity = None
        self.stiffness_reduction = None
        self.ply_ids = None
        self.ply_angles = None  # Per-element ply orientation angles (degrees)
        self.void_elements = None

        # Resolve the ply_angles sentinel (#44 item 2). ``None`` is the
        # deprecated path and emits a DeprecationWarning inside
        # ``_resolve_ply_angles``.
        self._input_ply_angles = _resolve_ply_angles(
            ply_angles, none_means='QI', caller='CompositeMesh.ply_angles')
        self.generate_mesh()

    def generate_mesh(self):
        x = np.linspace(0, self.L_x, self.nx + 1)
        y = np.linspace(0, self.L_y, self.ny + 1)
        z = np.linspace(0, self.L_z, self.nz + 1)

        nodes = []

        for zk in z:
            for yj in y:
                for xi in x:
                    nodes.append([xi, yj, zk])

        self.nodes = np.array(nodes)

        # Sample porosity at all nodes
        self.porosity = self.porosity_field.local_porosity(
            self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2])
        self.stiffness_reduction = self.porosity_field.local_stiffness_reduction(
            self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2])

        # Ply IDs
        z_normalized = self.nodes[:, 2] / self.L_z
        self.ply_ids = np.clip((z_normalized * self.material.n_plies).astype(int),
                               0, self.material.n_plies - 1)

        # Hex element connectivity
        elements = []
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    n0 = k * (self.ny + 1) * (self.nx + 1) + j * (self.nx + 1) + i
                    n1 = n0 + 1
                    n2 = n0 + (self.nx + 1) + 1
                    n3 = n0 + (self.nx + 1)
                    n4 = n0 + (self.ny + 1) * (self.nx + 1)
                    n5 = n4 + 1
                    n6 = n4 + (self.nx + 1) + 1
                    n7 = n4 + (self.nx + 1)
                    elements.append([n0, n1, n2, n3, n4, n5, n6, n7])

        self.elements = np.array(elements)

        # Identify void elements: check if element centroid falls inside
        # any discrete void geometry (explicit inclusion modeling)
        elem_centers = np.mean(self.nodes[self.elements], axis=1)  # (n_elem, 3)
        void_mask = np.zeros(len(self.elements), dtype=bool)
        for void in self.porosity_field.discrete_voids:
            inside = void.contains(elem_centers[:, 0], elem_centers[:, 1], elem_centers[:, 2])
            void_mask |= inside
        self.void_elements = np.where(void_mask)[0]
        # Also create a set for O(1) lookup
        self.void_element_set = set(self.void_elements.tolist())

        # Assign per-element ply angles (degrees)
        # Element ply_id is determined by the centroid z-coordinate
        elem_centroids_z = np.mean(self.nodes[self.elements][:, :, 2], axis=1)
        elem_ply_ids = np.clip(
            (elem_centroids_z / self.L_z * self.material.n_plies).astype(int),
            0, self.material.n_plies - 1)
        self.elem_ply_ids = elem_ply_ids

        if self._input_ply_angles is not None:
            angle_list = list(self._input_ply_angles)
            if len(angle_list) < self.material.n_plies:
                # Repeat to fill all plies
                angle_list = (angle_list * (self.material.n_plies // len(angle_list) + 1))[:self.material.n_plies]
            self.ply_angles = np.array([angle_list[pid] for pid in elem_ply_ids], dtype=float)
        else:
            # Default: all 0-degree plies
            self.ply_angles = np.zeros(len(self.elements), dtype=float)

        logger.info("Mesh generated: %d nodes, %d elements",
                    len(self.nodes), len(self.elements))
        logger.info("  Domain: %.1f x %.1f x %.2f mm",
                    self.L_x, self.L_y, self.L_z)
        logger.info("  Void elements: %d", len(self.void_elements))

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def n_dof(self) -> int:
        return self.n_nodes * 3

    @property
    def domain_size(self) -> Tuple[float, float, float]:
        return (self.L_x, self.L_y, self.L_z)

    def nodes_on_face(self, face: str) -> np.ndarray:
        """Return node indices on the specified face.

        Parameters
        ----------
        face : str
            One of 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'.

        Returns
        -------
        np.ndarray
            1-D array of node indices on that face.
        """
        tol = 1e-8
        coords = self.nodes
        if face == 'x_min':
            return np.where(np.abs(coords[:, 0] - coords[:, 0].min()) < tol)[0]
        elif face == 'x_max':
            return np.where(np.abs(coords[:, 0] - coords[:, 0].max()) < tol)[0]
        elif face == 'y_min':
            return np.where(np.abs(coords[:, 1] - coords[:, 1].min()) < tol)[0]
        elif face == 'y_max':
            return np.where(np.abs(coords[:, 1] - coords[:, 1].max()) < tol)[0]
        elif face == 'z_min':
            return np.where(np.abs(coords[:, 2] - coords[:, 2].min()) < tol)[0]
        elif face == 'z_max':
            return np.where(np.abs(coords[:, 2] - coords[:, 2].max()) < tol)[0]
        else:
            raise ValueError(f"Unknown face '{face}'. Use x_min/x_max/y_min/y_max/z_min/z_max.")

    def find_nodes_near(self, x: Optional[float] = None,
                        y: Optional[float] = None,
                        z: Optional[float] = None,
                        tol: Optional[float] = None) -> np.ndarray:
        """Return node indices within ``tol`` of the specified target coords.

        Any of ``x``, ``y``, ``z`` may be ``None``, in which case that axis
        is not used in the distance computation (i.e. the search becomes a
        line/plane match rather than a point match). Distances are computed
        with ``np.linalg.norm`` on the subset of axes that were specified.

        Parameters
        ----------
        x, y, z : float or None
            Target coordinate per axis. Pass ``None`` to ignore an axis.
        tol : float or None
            Distance tolerance. If ``None``, defaults to half of a typical
            element edge length (``0.5 * min(L_x/nx, L_y/ny, L_z/nz)``).

        Returns
        -------
        np.ndarray
            Sorted 1-D array of node indices whose distance to the target
            (restricted to the specified axes) is ``<= tol``.

        Notes
        -----
        Used by ILSS short-beam BCs to locate midspan-top loading nodes
        even when ``Lx / 2`` does not coincide with a mesh node.
        """
        if x is None and y is None and z is None:
            raise ValueError(
                "find_nodes_near: at least one of x/y/z must be specified."
            )
        if tol is None:
            dxs = []
            if self.nx > 0:
                dxs.append(self.L_x / self.nx)
            if self.ny > 0:
                dxs.append(self.L_y / self.ny)
            if self.nz > 0:
                dxs.append(self.L_z / self.nz)
            tol = 0.5 * min(dxs)

        coords = self.nodes
        targets = []
        cols = []
        if x is not None:
            targets.append(float(x))
            cols.append(0)
        if y is not None:
            targets.append(float(y))
            cols.append(1)
        if z is not None:
            targets.append(float(z))
            cols.append(2)

        diffs = coords[:, cols] - np.asarray(targets, dtype=float)
        dist = np.linalg.norm(diffs, axis=1)
        return np.where(dist <= tol)[0]

    def __repr__(self) -> str:
        return (f"CompositeMesh(nx={self.nx}, ny={self.ny}, nz={self.nz}, "
                f"n_nodes={self.n_nodes}, n_elements={self.n_elements}, "
                f"domain={self.L_x:.1f}x{self.L_y:.1f}x{self.L_z:.2f}mm, "
                f"void_elements={len(self.void_elements)})")


def check_mesh_quality(mesh: CompositeMesh, verbose: bool = False) -> Dict:
    """Check mesh quality: element aspect ratios and Jacobian determinants.

    Parameters
    ----------
    mesh : CompositeMesh
        The finite element mesh to check.
    verbose : bool
        Print detailed quality report.

    Returns
    -------
    dict
        Quality metrics: min/max aspect ratio, min Jacobian determinant,
        number of inverted elements, number of highly distorted elements.

    Raises
    ------
    Warning messages are printed for inverted or highly distorted elements.
    """
    # Lazy import — :class:`Hex8Element` lives in :mod:`porosity_fe.fe.element`
    # which is one layer above us in the dependency graph (the FE subpackage
    # imports from :mod:`porosity_fe.mesh`, not the other way around). Importing
    # here keeps :mod:`porosity_fe.mesh` cycle-free at module load time.
    from .fe.element import Hex8Element

    n_elem = mesh.n_elements
    aspect_ratios = np.empty(n_elem)
    min_detJ_per_elem = np.empty(n_elem)

    for e in range(n_elem):
        node_ids = mesh.elements[e]
        coords = mesh.nodes[node_ids]  # (8, 3)

        # Aspect ratio: ratio of max edge length to min edge length
        # Check all 12 edges of a hexahedron
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
        ]
        edge_lengths = np.array([np.linalg.norm(coords[a] - coords[b])
                                  for a, b in edges])
        min_len = edge_lengths.min()
        max_len = edge_lengths.max()
        aspect_ratios[e] = max_len / min_len if min_len > 1e-15 else np.inf

        # Jacobian at element center
        dN = Hex8Element.shape_derivatives(0.0, 0.0, 0.0)
        J = dN @ coords
        min_detJ_per_elem[e] = np.linalg.det(J)

    n_inverted = int(np.sum(min_detJ_per_elem < 0))
    n_distorted = int(np.sum(aspect_ratios > 20.0))

    result = {
        'min_aspect_ratio': float(np.min(aspect_ratios)),
        'max_aspect_ratio': float(np.max(aspect_ratios)),
        'mean_aspect_ratio': float(np.mean(aspect_ratios)),
        'min_jacobian_det': float(np.min(min_detJ_per_elem)),
        'n_inverted': n_inverted,
        'n_distorted': n_distorted,
        'n_elements': n_elem,
    }

    if verbose:
        logger.info("  Mesh quality: %d elements", n_elem)
        logger.info(
            "    Aspect ratio: min=%.2f, max=%.2f, mean=%.2f",
            result['min_aspect_ratio'],
            result['max_aspect_ratio'],
            result['mean_aspect_ratio'],
        )
        logger.info("    Min Jacobian det: %.6e", result['min_jacobian_det'])
        if n_inverted > 0:
            logger.warning(
                "    WARNING: %d inverted elements (negative Jacobian)!",
                n_inverted,
            )
        if n_distorted > 0:
            logger.warning(
                "    WARNING: %d highly distorted elements (aspect ratio > 20)!",
                n_distorted,
            )

    if n_inverted > 0:
        warnings.warn(
            f"Mesh has {n_inverted} inverted elements (negative Jacobian determinant).",
            stacklevel=2,
        )
    if n_distorted > 0:
        warnings.warn(
            f"Mesh has {n_distorted} highly distorted elements (aspect ratio > 20).",
            stacklevel=2,
        )

    return result

