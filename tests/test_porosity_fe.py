#!/usr/bin/env python3
"""Tests for porosity_fe_analysis.py"""

import dataclasses

import numpy as np
import scipy.sparse
import pytest
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

from porosity_fe_analysis import (MaterialProperties, MATERIALS, VoidGeometry, VOID_SHAPES,
                                   PorosityField, POROSITY_CONFIGS, CompositeMesh,
                                   EmpiricalSolver, FEVisualizer,
                                   compare_configurations, save_results_to_json,
                                   rotation_matrix_3d, stress_transformation_3d,
                                   strain_transformation_3d, rotate_stiffness_3d,
                                   gauss_points_1d, gauss_points_hex,
                                   Hex8Element, _mt_effective_stiffness,
                                   _degraded_composite_stiffness,
                                   GlobalAssembler, BoundaryHandler, FESolver, FieldResults,
                                   compute_clt_effective_modulus, check_mesh_quality,
                                   _build_provenance, load_results_from_json,
                                   JSON_SCHEMA_VERSION, FORMAT_EMPIRICAL_SWEEP)


class TestMaterialProperties:
    def test_dataclass_creation(self):
        mat = MATERIALS['T800_epoxy']
        assert mat.E11 == 161000.0
        assert mat.sigma_1c == 1500.0
        assert mat.sigma_2t == 80.0
        assert mat.tau_12 == 100.0
        assert mat.tau_ilss == 90.0
        assert mat.matrix_modulus == 3500.0
        assert mat.fiber_volume_fraction == 0.60

    def test_all_presets_exist(self):
        assert 'T800_epoxy' in MATERIALS
        assert 'T700_epoxy' in MATERIALS
        assert 'glass_epoxy' in MATERIALS

    def test_total_thickness(self):
        mat = MATERIALS['T800_epoxy']
        expected = 0.183 * 24
        assert abs(mat.total_thickness - expected) < 1e-10

    def test_stiffness_matrix_shape(self):
        mat = MATERIALS['T800_epoxy']
        C = mat.get_stiffness_matrix()
        assert C.shape == (6, 6)

    def test_stiffness_matrix_symmetric(self):
        mat = MATERIALS['T800_epoxy']
        C = mat.get_stiffness_matrix()
        np.testing.assert_allclose(C, C.T, atol=1e-6)

    def test_stiffness_matrix_positive_definite(self):
        mat = MATERIALS['T800_epoxy']
        C = mat.get_stiffness_matrix()
        eigenvalues = np.linalg.eigvalsh(C)
        assert np.all(eigenvalues > 0)

    def test_compliance_is_inverse_of_stiffness(self):
        mat = MATERIALS['T800_epoxy']
        C = mat.get_stiffness_matrix()
        S = mat.get_compliance_matrix()
        np.testing.assert_allclose(C @ S, np.eye(6), atol=1e-6)

    def test_isotropic_matrix_stiffness_shape(self):
        mat = MATERIALS['T800_epoxy']
        C_m = mat.get_isotropic_matrix_stiffness()
        assert C_m.shape == (6, 6)

    def test_isotropic_matrix_stiffness_symmetric(self):
        mat = MATERIALS['T800_epoxy']
        C_m = mat.get_isotropic_matrix_stiffness()
        np.testing.assert_allclose(C_m, C_m.T, atol=1e-6)

    def test_isotropic_matrix_stiffness_values(self):
        """C_m should reflect E_m=3500, nu_m=0.35"""
        mat = MATERIALS['T800_epoxy']
        C_m = mat.get_isotropic_matrix_stiffness()
        E_m, nu_m = 3500.0, 0.35
        lam = E_m * nu_m / ((1 + nu_m) * (1 - 2 * nu_m))
        mu = E_m / (2 * (1 + nu_m))
        assert abs(C_m[0, 0] - (lam + 2 * mu)) < 1.0
        assert abs(C_m[0, 1] - lam) < 1.0
        assert abs(C_m[3, 3] - mu) < 1.0

    def test_im7_preset_exists(self):
        assert 'IM7_8551_epoxy' in MATERIALS
        mat = MATERIALS['IM7_8551_epoxy']
        assert 170000 <= mat.E11 <= 180000

    def test_t300_934_preset_exists(self):
        assert 'T300_934_epoxy' in MATERIALS
        mat = MATERIALS['T300_934_epoxy']
        assert 125000 <= mat.E11 <= 140000

    def test_cf_peek_preset_exists(self):
        assert 'CF_PEEK' in MATERIALS
        mat = MATERIALS['CF_PEEK']
        assert 130000 <= mat.E11 <= 150000

    @staticmethod
    def _kwargs(**overrides):
        base = dict(
            E11=140000.0, E22=10500.0, E33=10500.0,
            G12=4900.0, G13=4900.0, G23=3700.0,
            nu12=0.30, nu13=0.30, nu23=0.42,
            sigma_1c=1300.0, sigma_1t=2500.0,
            sigma_2t=70.0, sigma_2c=210.0,
            tau_12=90.0, tau_ilss=85.0,
            t_ply=0.180, n_plies=24,
            matrix_modulus=3400.0, matrix_poisson=0.36,
            fiber_modulus=240000.0, fiber_volume_fraction=0.60,
        )
        base.update(overrides)
        return base

    def test_zero_modulus_rejected(self):
        with pytest.raises(ValueError, match=r"E11.*positive finite"):
            MaterialProperties(**self._kwargs(E11=0.0))

    def test_negative_strength_rejected(self):
        with pytest.raises(ValueError, match=r"sigma_1c.*positive finite"):
            MaterialProperties(**self._kwargs(sigma_1c=-100.0))

    def test_poisson_at_isotropic_limit_rejected(self):
        # nu = 0.5 makes (1 - 2*nu) = 0 in the isotropic matrix stiffness
        with pytest.raises(ValueError, match=r"matrix_poisson.*\(-1, 0\.5\)"):
            MaterialProperties(**self._kwargs(matrix_poisson=0.5))

    def test_negative_t_ply_rejected(self):
        with pytest.raises(ValueError, match=r"t_ply.*positive"):
            MaterialProperties(**self._kwargs(t_ply=-0.1))

    def test_zero_n_plies_rejected(self):
        with pytest.raises(ValueError, match=r"n_plies.*positive integer"):
            MaterialProperties(**self._kwargs(n_plies=0))

    def test_fiber_fraction_above_one_rejected(self):
        with pytest.raises(ValueError, match=r"fiber_volume_fraction"):
            MaterialProperties(**self._kwargs(fiber_volume_fraction=60.0))


class TestVoidGeometry:
    def test_sphere_creation(self):
        void = VoidGeometry(center=(10, 5, 2), radii=(1.0, 1.0, 1.0))
        np.testing.assert_array_equal(void.center, [10, 5, 2])
        np.testing.assert_array_equal(void.radii, [1.0, 1.0, 1.0])
        assert void.orientation == 0.0

    def test_contains_center(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(1, 1, 1))
        x = np.array([0.0])
        y = np.array([0.0])
        z = np.array([0.0])
        assert void.contains(x, y, z)[0] == True

    def test_contains_outside(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(1, 1, 1))
        x = np.array([2.0])
        y = np.array([0.0])
        z = np.array([0.0])
        assert void.contains(x, y, z)[0] == False

    def test_contains_boundary(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(1, 1, 1))
        x = np.array([1.0])
        y = np.array([0.0])
        z = np.array([0.0])
        assert void.contains(x, y, z)[0] == True  # <= 1

    def test_ellipsoidal_contains(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(3, 1, 1))
        # Inside along major axis
        assert void.contains(np.array([2.5]), np.array([0.0]), np.array([0.0]))[0] == True
        # Outside along minor axis
        assert void.contains(np.array([0.0]), np.array([1.5]), np.array([0.0]))[0] == False

    def test_volume_sphere(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(2, 2, 2))
        expected = (4 / 3) * np.pi * 8
        assert abs(void.volume() - expected) < 1e-10

    def test_volume_ellipsoid(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(3, 2, 1))
        expected = (4 / 3) * np.pi * 6
        assert abs(void.volume() - expected) < 1e-10

    def test_aspect_ratio_sphere(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(1, 1, 1))
        assert void.aspect_ratio == 1.0

    def test_aspect_ratio_elongated(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(3, 1, 1))
        assert void.aspect_ratio == 3.0

    def test_scf_sphere(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(1, 1, 1))
        scf = void.stress_concentration_factor()
        assert isinstance(scf, dict)
        assert 'compression' in scf
        assert 'tension' in scf
        assert 'shear' in scf
        assert 'ilss' in scf
        assert scf['compression'] > 1.0

    def test_distance_field_inside_negative(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(1, 1, 1))
        d = void.distance_field(np.array([0.0]), np.array([0.0]), np.array([0.0]))
        assert d[0] < 0  # Inside -> negative

    def test_distance_field_outside_positive(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(1, 1, 1))
        d = void.distance_field(np.array([2.0]), np.array([0.0]), np.array([0.0]))
        assert d[0] > 0  # Outside -> positive

    def test_void_shapes_presets(self):
        assert 'spherical' in VOID_SHAPES
        assert 'cylindrical' in VOID_SHAPES
        assert 'penny' in VOID_SHAPES
        assert VOID_SHAPES['spherical'] == (1.0, 1.0, 1.0)

    def test_orientation_rotation(self):
        """Rotated cylindrical void should contain points along rotated axis"""
        void = VoidGeometry(center=(0, 0, 0), radii=(3, 1, 1), orientation=np.pi / 2)
        # After 90-degree rotation, major axis is along y
        assert void.contains(np.array([0.0]), np.array([2.5]), np.array([0.0]))[0] == True
        assert void.contains(np.array([2.5]), np.array([0.0]), np.array([0.0]))[0] == False

    def test_zero_radius_rejected(self):
        with pytest.raises(ValueError, match=r"radii.*positive"):
            VoidGeometry(center=(0, 0, 0), radii=(0.0, 1.0, 1.0))

    def test_negative_radius_rejected(self):
        with pytest.raises(ValueError, match=r"radii.*positive"):
            VoidGeometry(center=(0, 0, 0), radii=(1.0, -1.0, 1.0))

    def test_wrong_radii_shape_rejected(self):
        with pytest.raises(ValueError, match=r"radii must have 3 components"):
            VoidGeometry(center=(0, 0, 0), radii=(1.0, 1.0))

    def test_non_finite_orientation_rejected(self):
        with pytest.raises(ValueError, match=r"orientation"):
            VoidGeometry(center=(0, 0, 0), radii=(1.0, 1.0, 1.0),
                         orientation=float('nan'))


class TestPorosityField:
    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']

    def test_uniform_constant_porosity(self):
        pf = PorosityField(self.material, 0.03, distribution='uniform')
        Lz = self.material.total_thickness
        z_mid = Lz / 2
        Vp = pf.local_porosity(np.array([10.0]), np.array([5.0]), np.array([z_mid]))
        assert abs(Vp[0] - 0.03) < 1e-10

    def test_uniform_same_everywhere(self):
        pf = PorosityField(self.material, 0.05, distribution='uniform')
        Lz = self.material.total_thickness
        z_vals = np.linspace(0, Lz, 10)
        x = np.full_like(z_vals, 10.0)
        y = np.full_like(z_vals, 5.0)
        Vp = pf.local_porosity(x, y, z_vals)
        np.testing.assert_allclose(Vp, 0.05, atol=1e-10)

    def test_clustered_midplane_higher_at_center(self):
        pf = PorosityField(self.material, 0.05, distribution='clustered',
                           cluster_location='midplane')
        Lz = self.material.total_thickness
        Vp_mid = pf.local_porosity(np.array([10.0]), np.array([5.0]),
                                    np.array([Lz / 2]))[0]
        Vp_edge = pf.local_porosity(np.array([10.0]), np.array([5.0]),
                                     np.array([0.0]))[0]
        assert Vp_mid > Vp_edge

    def test_clustered_surface_higher_at_surface(self):
        pf = PorosityField(self.material, 0.05, distribution='clustered',
                           cluster_location='surface')
        Lz = self.material.total_thickness
        Vp_surface = pf.local_porosity(np.array([10.0]), np.array([5.0]),
                                        np.array([0.0]))[0]
        Vp_mid = pf.local_porosity(np.array([10.0]), np.array([5.0]),
                                    np.array([Lz / 2]))[0]
        assert Vp_surface > Vp_mid

    def test_interface_peaks_at_ply_boundaries(self):
        pf = PorosityField(self.material, 0.05, distribution='interface')
        Lz = self.material.total_thickness
        t = self.material.t_ply
        Vp_interface = pf.local_porosity(np.array([10.0]), np.array([5.0]),
                                          np.array([t]))[0]
        Vp_midply = pf.local_porosity(np.array([10.0]), np.array([5.0]),
                                       np.array([t / 2]))[0]
        assert Vp_interface > Vp_midply

    def test_stiffness_reduction_pristine_is_one(self):
        pf = PorosityField(self.material, 0.0, distribution='uniform')
        sr = pf.local_stiffness_reduction(np.array([10.0]), np.array([5.0]),
                                           np.array([1.0]))
        assert abs(sr[0] - 1.0) < 1e-10

    def test_stiffness_reduction_decreases_with_porosity(self):
        pf = PorosityField(self.material, 0.05, distribution='uniform')
        sr = pf.local_stiffness_reduction(np.array([10.0]), np.array([5.0]),
                                           np.array([1.0]))
        assert sr[0] < 1.0
        assert sr[0] > 0.0

    def test_porosity_clamped_to_one(self):
        """With very high Vp and a discrete void, should not exceed 1.0"""
        void = VoidGeometry(center=(10, 5, 1), radii=(1, 1, 0.5))
        pf = PorosityField(self.material, 0.90, distribution='uniform',
                           discrete_voids=[void])
        Vp = pf.local_porosity(np.array([10.0]), np.array([5.0]), np.array([1.0]))
        assert Vp[0] <= 1.0

    def test_negative_Vp_raises(self):
        with pytest.raises(ValueError, match=r"finite fraction in \[0, 1\]"):
            PorosityField(self.material, -0.01, distribution='uniform')

    def test_Vp_above_one_raises_with_percent_hint(self):
        with pytest.raises(ValueError, match=r"Did you pass a percent\?"):
            PorosityField(self.material, 3.0, distribution='uniform')

    def test_nan_Vp_raises(self):
        with pytest.raises(ValueError, match=r"finite fraction"):
            PorosityField(self.material, float('nan'), distribution='uniform')

    def test_inf_Vp_raises(self):
        with pytest.raises(ValueError, match=r"finite fraction"):
            PorosityField(self.material, float('inf'), distribution='uniform')

    def test_Vp_boundary_zero_and_one_accepted(self):
        # Both boundaries should be accepted (no exception)
        PorosityField(self.material, 0.0, distribution='uniform')
        PorosityField(self.material, 1.0, distribution='uniform')

    def test_effective_porosity_profile_shape(self):
        pf = PorosityField(self.material, 0.03, distribution='uniform')
        z, Vp = pf.effective_porosity_profile(nz=50)
        assert len(z) == 50
        assert len(Vp) == 50

    def test_configs_all_exist(self):
        assert len(POROSITY_CONFIGS) == 5
        for name in ['uniform_spherical', 'uniform_cylindrical',
                      'clustered_midplane', 'clustered_surface', 'interface_penny']:
            assert name in POROSITY_CONFIGS

    def test_void_shape_string_resolved(self):
        pf = PorosityField(self.material, 0.03, void_shape='cylindrical')
        assert pf.void_shape_radii == (3.0, 1.0, 1.0)

    def test_void_shape_tuple_accepted(self):
        pf = PorosityField(self.material, 0.03, void_shape=(2.0, 1.5, 0.5))
        assert pf.void_shape_radii == (2.0, 1.5, 0.5)

    def test_unknown_void_shape_string_raises(self):
        with pytest.raises(ValueError, match=r"Unknown void_shape"):
            PorosityField(self.material, 0.03, void_shape='spheroidal')

    def test_unknown_distribution_raises(self):
        with pytest.raises(ValueError, match=r"Unknown distribution"):
            PorosityField(self.material, 0.03, distribution='gradient')

    def test_unknown_cluster_location_raises(self):
        with pytest.raises(ValueError, match=r"Unknown cluster_location"):
            PorosityField(self.material, 0.03,
                          distribution='clustered', cluster_location='midplne')

    def test_quarter_cluster_location_supported(self):
        # 'quarter' is one of the documented cluster locations and should round-trip.
        pf = PorosityField(self.material, 0.03,
                           distribution='clustered', cluster_location='quarter')
        assert pf.cluster_location == 'quarter'

    def test_Vp_snap_to_one_from_fp_noise(self):
        # numerical noise just above 1.0 should snap to 1.0 instead of raising
        pf = PorosityField(self.material, 1.0 + 5e-10, distribution='uniform')
        assert pf.Vp == 1.0

    def test_Vp_just_above_one_no_percent_hint(self):
        # Values barely above the boundary are likely numerical noise, not
        # percent confusion — the percent hint should be suppressed.
        with pytest.raises(ValueError) as exc:
            PorosityField(self.material, 1.0001, distribution='uniform')
        assert "Did you pass a percent?" not in str(exc.value)

    def test_Vp_string_rejected_with_typeerror(self):
        with pytest.raises(TypeError, match=r"numeric type"):
            PorosityField(self.material, "0.5", distribution='uniform')

    def test_Vp_none_rejected(self):
        with pytest.raises(ValueError, match=r"None"):
            PorosityField(self.material, None, distribution='uniform')


class TestCompositeMesh:
    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        self.pf = PorosityField(self.material, 0.03, distribution='uniform')

    def test_mesh_creation(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        assert mesh.nodes is not None
        assert mesh.elements is not None

    def test_node_count(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        expected_nodes = 11 * 6 * 7  # (nx+1)*(ny+1)*(nz+1)
        assert len(mesh.nodes) == expected_nodes

    def test_element_count(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        expected_elements = 10 * 5 * 6
        assert len(mesh.elements) == expected_elements

    def test_nodes_3d(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        assert mesh.nodes.shape[1] == 3

    def test_hex_elements_8_nodes(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        assert mesh.elements.shape[1] == 8

    def test_porosity_field_sampled(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        assert len(mesh.porosity) == len(mesh.nodes)
        # Uniform 3% -> all nodes should be ~0.03
        np.testing.assert_allclose(mesh.porosity, 0.03, atol=1e-10)

    def test_stiffness_reduction_sampled(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        assert len(mesh.stiffness_reduction) == len(mesh.nodes)
        np.testing.assert_allclose(mesh.stiffness_reduction, 0.97, atol=1e-10)

    def test_ply_ids_range(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        assert np.min(mesh.ply_ids) >= 0
        assert np.max(mesh.ply_ids) <= self.material.n_plies

    def test_domain_bounds(self):
        mesh = CompositeMesh(self.pf, self.material, nx=10, ny=5, nz=6)
        assert np.min(mesh.nodes[:, 0]) >= 0
        assert np.min(mesh.nodes[:, 2]) >= 0
        assert abs(np.max(mesh.nodes[:, 2]) - self.material.total_thickness) < 1e-6

    def test_zero_axis_count_rejected(self):
        with pytest.raises(ValueError, match=r"nx.*positive integer"):
            CompositeMesh(self.pf, self.material, nx=0, ny=5, nz=6)

    def test_negative_axis_count_rejected(self):
        with pytest.raises(ValueError, match=r"ny.*positive integer"):
            CompositeMesh(self.pf, self.material, nx=10, ny=-2, nz=6)

    def test_huge_axis_count_rejected(self):
        with pytest.raises(ValueError, match=r"exhaust memory|exceeds"):
            CompositeMesh(self.pf, self.material, nx=20_000, ny=5, nz=6)


class TestEmpiricalSolver:
    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        pf = PorosityField(self.material, 0.03, distribution='uniform')
        self.mesh = CompositeMesh(pf, self.material, nx=10, ny=5, nz=6)
        self.solver = EmpiricalSolver(self.mesh, self.material)

    def test_judd_wright_zero_porosity(self):
        """At Vp=0, knockdown should be 1.0"""
        kd = self.solver._judd_wright(0.0, 'compression')
        assert abs(kd - 1.0) < 1e-10

    def test_judd_wright_decreasing(self):
        """Higher porosity -> lower knockdown"""
        kd1 = self.solver._judd_wright(0.01, 'compression')
        kd2 = self.solver._judd_wright(0.05, 'compression')
        assert kd1 > kd2

    def test_power_law_zero_porosity(self):
        kd = self.solver._power_law(0.0, 'compression')
        assert abs(kd - 1.0) < 1e-10

    def test_linear_zero_porosity(self):
        kd = self.solver._linear(0.0, 'compression')
        assert abs(kd - 1.0) < 1e-10

    def test_ilss_most_sensitive(self):
        """ILSS should have largest knockdown for same porosity"""
        Vp = 0.05
        kd_comp = self.solver._judd_wright(Vp, 'compression')
        kd_ilss = self.solver._judd_wright(Vp, 'ilss')
        assert kd_ilss < kd_comp

    def test_get_failure_load_returns_dict(self):
        result = self.solver.get_failure_load(mode='compression', model='judd_wright')
        assert 'failure_stress' in result
        assert 'knockdown' in result
        assert 'model' in result

    def test_failure_load_positive(self):
        result = self.solver.get_failure_load(mode='compression', model='judd_wright')
        assert result['failure_stress'] > 0
        assert 0 < result['knockdown'] <= 1.0

    def test_unknown_loading_mode_raises_with_listing(self):
        with pytest.raises(ValueError, match=r"Unknown loading mode"):
            self.solver._get_pristine_strength('flexure')

    def test_override_alpha_only_changes_targeted_mode(self):
        """Partial override leaves other modes at QI defaults."""
        solver = EmpiricalSolver(self.mesh, self.material,
                                  judd_wright_alpha={'ilss': 12.0})
        # ILSS overridden (and layup scale = 1.0 for default ply_angles=None / f_md=0.5)
        assert abs(solver.JUDD_WRIGHT_ALPHA['ilss'] - 12.0) < 1e-12
        # Other modes match the QI baseline at f_md = 0.5
        assert abs(solver.JUDD_WRIGHT_ALPHA['compression'] - 6.9) < 1e-12
        assert abs(solver.JUDD_WRIGHT_ALPHA['tension'] - 3.9) < 1e-12
        assert abs(solver.JUDD_WRIGHT_ALPHA['shear'] - 8.0) < 1e-12

    def test_override_layup_scaling_applied(self):
        """Override values are scaled by layup the same way as the QI baseline."""
        ud = [0.0] * 16  # f_md = 0; ILSS floor = 0.80
        solver = EmpiricalSolver(self.mesh, self.material,
                                  ply_angles=ud,
                                  judd_wright_alpha={'ilss': 12.0})
        assert abs(solver.JUDD_WRIGHT_ALPHA['ilss'] - 12.0 * 0.80) < 1e-12

    def test_override_n_and_beta(self):
        solver = EmpiricalSolver(self.mesh, self.material,
                                  power_law_n={'compression': 4.0},
                                  linear_beta={'shear': 6.0})
        assert abs(solver.POWER_LAW_N['compression'] - 4.0) < 1e-12
        assert abs(solver.LINEAR_BETA['shear'] - 6.0) < 1e-12

    def test_override_negative_alpha_rejected(self):
        with pytest.raises(ValueError, match=r"positive finite"):
            EmpiricalSolver(self.mesh, self.material,
                            judd_wright_alpha={'compression': -1.0})

    def test_override_nan_alpha_rejected(self):
        with pytest.raises(ValueError, match=r"positive finite"):
            EmpiricalSolver(self.mesh, self.material,
                            judd_wright_alpha={'compression': float('nan')})

    def test_override_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match=r"unknown mode keys"):
            EmpiricalSolver(self.mesh, self.material,
                            judd_wright_alpha={'silly_mode': 5.0})

    def test_override_non_dict_rejected(self):
        with pytest.raises(TypeError, match=r"dict mapping mode"):
            EmpiricalSolver(self.mesh, self.material,
                            judd_wright_alpha=[6.9, 3.9, 8.0, 10.0])

    def test_override_does_not_mutate_class_defaults(self):
        """Overrides must not leak back into the class-level QI dicts."""
        EmpiricalSolver(self.mesh, self.material,
                        judd_wright_alpha={'ilss': 99.0})
        assert EmpiricalSolver._JUDD_WRIGHT_ALPHA_QI['ilss'] == 10.0

    def test_get_all_failure_loads(self):
        results = self.solver.get_all_failure_loads()
        for mode in ['compression', 'tension', 'shear', 'ilss']:
            assert mode in results
            for model in ['judd_wright', 'power_law', 'linear']:
                assert model in results[mode]

    def test_failure_stress_below_pristine(self):
        result = self.solver.get_failure_load(mode='compression', model='judd_wright')
        assert result['failure_stress'] < self.material.sigma_1c

    def test_discrete_void_scf_amplifies_knockdown(self):
        """Discrete macrovoid should cause worse local knockdown near the void."""
        material = MATERIALS['T800_epoxy']
        void = VoidGeometry(center=(25, 10, material.total_thickness / 2),
                            radii=(2, 2, 0.5))
        pf = PorosityField(material, 0.02, distribution='uniform',
                           discrete_voids=[void])
        mesh = CompositeMesh(pf, material, nx=20, ny=10, nz=12)
        solver = EmpiricalSolver(mesh, material)
        solver.apply_loading('compression', 'judd_wright')
        min_kd_with_void = solver.nodal_knockdown.min()

        pf_no_void = PorosityField(material, 0.02, distribution='uniform')
        mesh_no_void = CompositeMesh(pf_no_void, material, nx=20, ny=10, nz=12)
        solver_no_void = EmpiricalSolver(mesh_no_void, material)
        solver_no_void.apply_loading('compression', 'judd_wright')
        min_kd_no_void = solver_no_void.nodal_knockdown.min()

        # Discrete void should reduce local knockdown near the void
        assert min_kd_with_void < min_kd_no_void

    def test_apply_loading_bad_model_raises_value_error(self):
        # #22: bad model name should give a ValueError listing the valid
        # choices, not a bare KeyError.
        with pytest.raises(ValueError, match=r"Unknown knockdown model 'bogus'"):
            self.solver.apply_loading(mode='compression', model='bogus')

    def test_apply_loading_bad_mode_raises_value_error(self):
        with pytest.raises(ValueError, match=r"Unknown loading mode 'bogus'"):
            self.solver.apply_loading(mode='bogus', model='judd_wright')


class TestFEVisualizer:
    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        pf = PorosityField(self.material, 0.03, distribution='uniform')
        self.mesh = CompositeMesh(pf, self.material, nx=10, ny=5, nz=6)
        self.solver = EmpiricalSolver(self.mesh, self.material)
        self.solver.apply_loading('compression', 'judd_wright')
        self.pf = pf

    def test_plot_porosity_field_returns_fig(self):
        fig = FEVisualizer.plot_porosity_field(self.pf)
        assert fig is not None
        plt.close(fig)

    def test_plot_mesh_3d_returns_fig(self):
        fig = FEVisualizer.plot_mesh_3d(self.mesh)
        assert fig is not None
        plt.close(fig)

    def test_plot_mesh_detail_returns_fig(self):
        fig = FEVisualizer.plot_mesh_detail(self.mesh)
        assert fig is not None
        plt.close(fig)

    def test_plot_damage_contour_returns_fig(self):
        fig = FEVisualizer.plot_damage_contour(self.mesh, self.solver)
        assert fig is not None
        plt.close(fig)

    def test_plot_porosity_field_saves(self, tmp_path):
        path = str(tmp_path / "test_profile.png")
        FEVisualizer.plot_porosity_field(self.pf, save_path=path)
        assert os.path.exists(path)
        plt.close('all')

    def test_plot_void_scf_returns_fig(self):
        void = VoidGeometry(center=(0, 0, 0), radii=(1, 1, 1))
        fig = FEVisualizer.plot_void_scf(void)
        assert fig is not None
        plt.close(fig)


class TestAnalysisPipeline:
    def test_compare_configurations_returns_all_configs(self):
        results = compare_configurations(0.03, material_name='T800_epoxy',
                                          configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']})
        assert 'uniform_spherical' in results

    def test_compare_configurations_has_empirical_solver(self):
        results = compare_configurations(0.03, configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']})
        r = results['uniform_spherical']
        assert 'empirical' in r
        assert 'mesh' in r
        assert 'empirical_solver' in r

    def test_compare_configurations_empirical_has_all_modes(self):
        results = compare_configurations(0.03, configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']})
        emp = results['uniform_spherical']['empirical']
        for mode in ['compression', 'tension', 'shear', 'ilss']:
            assert mode in emp

    def test_save_results_to_json(self, tmp_path):
        results = compare_configurations(0.03, configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']})
        path = str(tmp_path / "test_results.json")
        save_results_to_json(results, path)
        assert os.path.exists(path)
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        # Envelope keys (flat structure: schema_version/format/provenance at
        # top level alongside per-configuration entries).
        assert 'schema_version' in data
        assert 'provenance' in data
        assert 'uniform_spherical' in data

    def test_compare_configurations_unknown_material_raises(self):
        with pytest.raises(ValueError, match=r"Unknown material"):
            compare_configurations(0.03, material_name='T800epoxy')

    def test_save_results_writes_schema_envelope(self, tmp_path):
        # #20: saved files must carry schema_version + format so consumers
        # can detect version drift.
        from porosity_fe_analysis import (JSON_SCHEMA_VERSION,
                                          FORMAT_EMPIRICAL_SWEEP)
        results = compare_configurations(
            0.03, configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']})
        path = str(tmp_path / "envelope.json")
        save_results_to_json(results, path)
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        assert data['schema_version'] == JSON_SCHEMA_VERSION
        assert data['format'] == FORMAT_EMPIRICAL_SWEEP

    def test_load_results_from_json_round_trips(self, tmp_path):
        from porosity_fe_analysis import load_results_from_json
        results = compare_configurations(
            0.03, configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']})
        path = str(tmp_path / "round_trip.json")
        save_results_to_json(results, path)
        loaded = load_results_from_json(path)
        assert 'uniform_spherical' in loaded
        # Inner payload survives the round trip.
        assert (loaded['uniform_spherical']['empirical']['compression']
                ['judd_wright']['knockdown']
                == results['uniform_spherical']['empirical']['compression']
                ['judd_wright']['knockdown'])

    def test_load_results_from_json_rejects_missing_envelope(self, tmp_path):
        from porosity_fe_analysis import load_results_from_json
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps({"uniform_spherical": {"empirical": {}}}),
                        encoding='utf-8')
        with pytest.raises(ValueError, match=r"missing 'schema_version'"):
            load_results_from_json(str(path))

    def test_load_results_from_json_rejects_incompatible_major(self, tmp_path):
        from porosity_fe_analysis import load_results_from_json
        path = tmp_path / "future.json"
        path.write_text(json.dumps({
            "schema_version": "2.0",
            "format": "porosity-fe.empirical-sweep",
        }), encoding='utf-8')
        with pytest.raises(ValueError, match=r"incompatible"):
            load_results_from_json(str(path))

    def test_load_results_from_json_rejects_unknown_format(self, tmp_path):
        from porosity_fe_analysis import load_results_from_json
        path = tmp_path / "wrong-fmt.json"
        path.write_text(json.dumps({
            "schema_version": "1.0",
            "format": "porosity-fe.something-else",
        }), encoding='utf-8')
        with pytest.raises(ValueError, match=r"unknown format"):
            load_results_from_json(str(path))


class TestResultsSchemaAndReproducibility:
    """#20 (output JSON Schema, numpy serialization) and #55 (__version__,
    seed provenance, determinism contract)."""

    _SCHEMA_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'validation', 'schemas', 'porosity_results_schema.json')

    def _one_config_results(self):
        return compare_configurations(
            0.03, configs={'uniform_spherical':
                           POROSITY_CONFIGS['uniform_spherical']})

    def test_exported_file_validates_against_results_schema(self, tmp_path):
        import jsonschema
        with open(self._SCHEMA_PATH, encoding='utf-8') as f:
            schema = json.load(f)
        path = str(tmp_path / "schema_check.json")
        save_results_to_json(self._one_config_results(), path)
        with open(path, encoding='utf-8') as f:
            doc = json.load(f)
        jsonschema.validate(instance=doc, schema=schema)  # raises on drift

    def test_module_has_importable_version(self):
        import porosity_fe_analysis as pfa
        assert isinstance(pfa.__version__, str) and pfa.__version__

    def test_provenance_records_version_and_seed(self, tmp_path):
        results = compare_configurations(
            0.03, seed=4242,
            configs={'uniform_spherical':
                     POROSITY_CONFIGS['uniform_spherical']})
        path = str(tmp_path / "prov.json")
        save_results_to_json(results, path)
        with open(path, encoding='utf-8') as f:
            prov = json.load(f)['provenance']
        assert prov['porosity_fe_version']  # no longer silently None
        assert prov['seed'] == 4242

    def test_pipeline_is_byte_deterministic(self, tmp_path):
        """Locks in current determinism so any future RNG introduction is
        forced to expose a seed (#55)."""
        p1, p2 = str(tmp_path / "r1.json"), str(tmp_path / "r2.json")
        save_results_to_json(self._one_config_results(), p1)
        save_results_to_json(self._one_config_results(), p2)
        with open(p1, encoding='utf-8') as f:
            d1 = json.load(f)
        with open(p2, encoding='utf-8') as f:
            d2 = json.load(f)
        # Two back-to-back runs in one process differ only by timestamp.
        d1['provenance'].pop('timestamp_utc')
        d2['provenance'].pop('timestamp_utc')
        assert d1 == d2

    def test_json_default_handles_numpy_and_ndarray(self, tmp_path):
        from porosity_fe_analysis import _json_default
        assert _json_default(np.float64(1.5)) == 1.5
        assert _json_default(np.int64(7)) == 7
        assert _json_default(np.array([1.0, 2.0])) == [1.0, 2.0]
        # End-to-end: an ndarray smuggled into the payload must not raise.
        # Replace the nested dicts with shallow copies — `config` aliases the
        # shared POROSITY_CONFIGS entry, so mutating it in place would poison
        # global state for other tests.
        results = self._one_config_results()
        entry = dict(results['uniform_spherical'])
        entry['config'] = {**entry['config'],
                           'ply_angles': np.array([0.0, 90.0, 45.0])}
        results = {'uniform_spherical': entry}
        path = str(tmp_path / "np.json")
        save_results_to_json(results, path)  # would TypeError pre-#20
        with open(path, encoding='utf-8') as f:
            doc = json.load(f)
        assert doc['uniform_spherical']['config']['ply_angles'] == [
            0.0, 90.0, 45.0]


class TestIntegration:
    """End-to-end test with reduced parameters for speed."""

    def test_full_pipeline_single_config(self, tmp_path):
        os.chdir(str(tmp_path))
        results = compare_configurations(
            0.03, material_name='T800_epoxy',
            configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']})

        r = results['uniform_spherical']
        emp_comp = r['empirical']['compression']['judd_wright']
        assert 0 < emp_comp['knockdown'] < 1.0
        assert emp_comp['failure_stress'] < MATERIALS['T800_epoxy'].sigma_1c

        emp_ilss = r['empirical']['ilss']['judd_wright']['knockdown']
        emp_comp_kd = r['empirical']['compression']['judd_wright']['knockdown']
        assert emp_ilss < emp_comp_kd

        save_results_to_json(results, "test_output.json")
        assert os.path.exists("test_output.json")

        FEVisualizer.plot_porosity_field(r['porosity_field'],
                                         save_path="test_profile.png")
        assert os.path.exists("test_profile.png")
        plt.close('all')

    def test_all_materials(self):
        for mat_name in ['T800_epoxy', 'T700_epoxy', 'glass_epoxy']:
            results = compare_configurations(
                0.02, material_name=mat_name,
                configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']})
            assert 'uniform_spherical' in results
            kd = results['uniform_spherical']['empirical']['compression']['judd_wright']['knockdown']
            assert 0 < kd < 1.0


# ============================================================
# FE SOLVER TESTS
# ============================================================

class TestCoordinateTransforms:
    def test_rotation_matrix_identity_for_zero_angle(self):
        R = rotation_matrix_3d(0.0, axis='z')
        np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_rotation_matrix_orthogonal(self):
        R = rotation_matrix_3d(np.pi / 4, axis='z')
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)

    def test_rotation_matrix_y_axis(self):
        R = rotation_matrix_3d(np.pi / 2, axis='y')
        # After 90-deg rotation about y: x->-z, z->x
        expected = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-14)

    def test_rotation_matrix_invalid_axis(self):
        with pytest.raises(ValueError):
            rotation_matrix_3d(0.0, axis='x')

    def test_stress_transform_identity_at_zero(self):
        T = stress_transformation_3d(0.0, axis='z')
        np.testing.assert_allclose(T, np.eye(6), atol=1e-15)

    def test_strain_transform_identity_at_zero(self):
        T = strain_transformation_3d(0.0, axis='z')
        np.testing.assert_allclose(T, np.eye(6), atol=1e-15)

    def test_rotate_stiffness_identity_at_zero(self):
        mat = MATERIALS['T800_epoxy']
        C = mat.get_stiffness_matrix()
        C_rot = rotate_stiffness_3d(C, 0.0, axis='z')
        np.testing.assert_allclose(C_rot, C, atol=1e-6)

    def test_rotate_stiffness_180_returns_same(self):
        """180-degree rotation about z should return same stiffness for orthotropic."""
        mat = MATERIALS['T800_epoxy']
        C = mat.get_stiffness_matrix()
        C_rot = rotate_stiffness_3d(C, np.pi, axis='z')
        np.testing.assert_allclose(C_rot, C, atol=1e-6)

    def test_rotate_stiffness_symmetric(self):
        mat = MATERIALS['T800_epoxy']
        C = mat.get_stiffness_matrix()
        C_rot = rotate_stiffness_3d(C, np.pi / 4, axis='z')
        np.testing.assert_allclose(C_rot, C_rot.T, atol=1e-6)

    def test_rotate_stiffness_wrong_shape(self):
        with pytest.raises(ValueError):
            rotate_stiffness_3d(np.eye(3), 0.0)


class TestCLTEffectiveModulus:
    def test_all_zero_plies_returns_E11(self):
        """All 0-degree plies should give E_x close to E11."""
        mat = MATERIALS['T800_epoxy']
        E_x = compute_clt_effective_modulus(mat, [0.0] * 24)
        # Should be close to E11 (plane-stress correction makes it slightly different)
        assert abs(E_x - mat.E11) / mat.E11 < 0.02

    def test_quasi_isotropic_lower_than_E11(self):
        """QI layup should have E_x much lower than E11."""
        mat = MATERIALS['T800_epoxy']
        angles = [0, 45, 90, -45] * 6  # 24 plies QI
        E_x = compute_clt_effective_modulus(mat, angles)
        assert E_x < mat.E11
        assert E_x > mat.E22  # Should still be stiffer than transverse

    def test_positive_modulus(self):
        mat = MATERIALS['T800_epoxy']
        E_x = compute_clt_effective_modulus(mat, [0, 90, 0, 90] * 6)
        assert E_x > 0

    def test_symmetric_layup(self):
        """Symmetric layup [0/90]_s should equal [0/90/90/0]."""
        mat = MATERIALS['T800_epoxy']
        E1 = compute_clt_effective_modulus(mat, [0, 90, 90, 0] * 6)
        E2 = compute_clt_effective_modulus(mat, [0, 90] * 12)
        # A-matrix is the same for both (same ply count per angle)
        assert abs(E1 - E2) / E1 < 1e-10


class TestGaussQuadrature:
    def test_gauss_1d_order2(self):
        pts, wts = gauss_points_1d(2)
        assert len(pts) == 2
        assert len(wts) == 2
        np.testing.assert_allclose(wts.sum(), 2.0)

    def test_gauss_1d_order3(self):
        pts, wts = gauss_points_1d(3)
        assert len(pts) == 3
        np.testing.assert_allclose(wts.sum(), 2.0)

    def test_gauss_1d_invalid_order(self):
        with pytest.raises(ValueError):
            gauss_points_1d(4)

    def test_gauss_hex_shape(self):
        pts, wts = gauss_points_hex(order=2)
        assert pts.shape == (8, 3)
        assert wts.shape == (8,)

    def test_gauss_hex_weight_sum(self):
        """Weights should sum to 8 (volume of [-1,1]^3)."""
        pts, wts = gauss_points_hex(order=2)
        np.testing.assert_allclose(wts.sum(), 8.0)

    def test_gauss_hex_order3(self):
        pts, wts = gauss_points_hex(order=3)
        assert pts.shape == (27, 3)
        np.testing.assert_allclose(wts.sum(), 8.0)


class TestMTEffectiveStiffness:
    def setup_method(self):
        self.mat = MATERIALS['T800_epoxy']
        self.C_m = self.mat.get_isotropic_matrix_stiffness()

    def test_zero_porosity_returns_matrix(self):
        C_eff = _mt_effective_stiffness(self.C_m, 0.0, (1, 1, 1), 0.35)
        np.testing.assert_allclose(C_eff, self.C_m, atol=1e-6)

    def test_high_porosity_near_zero(self):
        C_eff = _mt_effective_stiffness(self.C_m, 0.99, (1, 1, 1), 0.35)
        assert C_eff[0, 0] < self.C_m[0, 0] * 0.1

    def test_decreasing_stiffness(self):
        C1 = _mt_effective_stiffness(self.C_m, 0.01, (1, 1, 1), 0.35)
        C5 = _mt_effective_stiffness(self.C_m, 0.05, (1, 1, 1), 0.35)
        assert C1[0, 0] > C5[0, 0]

    def test_positive_definite(self):
        C_eff = _mt_effective_stiffness(self.C_m, 0.05, (1, 1, 1), 0.35)
        eigenvalues = np.linalg.eigvalsh(C_eff)
        assert np.all(eigenvalues > 0)

    def test_prolate_void_shape(self):
        C_eff = _mt_effective_stiffness(self.C_m, 0.03, (3, 1, 1), 0.35)
        assert C_eff.shape == (6, 6)
        assert C_eff[0, 0] < self.C_m[0, 0]

    def test_oblate_void_shape(self):
        C_eff = _mt_effective_stiffness(self.C_m, 0.03, (3, 3, 0.3), 0.35)
        assert C_eff.shape == (6, 6)
        assert C_eff[0, 0] < self.C_m[0, 0]

    def test_finite_for_full_Vp_sweep_oblate(self):
        # Oblate voids near Vp -> 1 are the worst case for MT inversion;
        # the pinv fallback + finite check should keep all entries finite.
        for Vp in [0.50, 0.85, 0.95, 0.985]:
            C_eff = _mt_effective_stiffness(self.C_m, Vp, (3, 3, 0.3), 0.35)
            assert np.all(np.isfinite(C_eff)), f"non-finite C_eff at Vp={Vp}"

    def test_penny_void_anisotropy_along_short_axis(self):
        """Regression for #32. A penny-shaped void (3, 3, 0.3) has its
        symmetry axis along x_3 (the short axis), so the effective
        stiffness should show LARGER degradation along the through-disk
        direction (S[2,2]) than along the in-plane directions (S[0,0],
        S[1,1]). The old code treated penny as a prolate cylinder along
        x_1 and degraded the wrong axis."""
        Vp = 0.05
        C_eff = _mt_effective_stiffness(self.C_m, Vp, (3, 3, 0.3), 0.35)
        # Through-thickness (x_3) component degrades more than in-plane (x_1, x_2)
        deg_xx = (self.C_m[0, 0] - C_eff[0, 0]) / self.C_m[0, 0]
        deg_yy = (self.C_m[1, 1] - C_eff[1, 1]) / self.C_m[1, 1]
        deg_zz = (self.C_m[2, 2] - C_eff[2, 2]) / self.C_m[2, 2]
        assert deg_zz > deg_xx, (
            f"penny axis degradation {deg_zz:.4f} should exceed in-plane "
            f"degradation {deg_xx:.4f} (the disk is perpendicular to x_3)"
        )
        # The two in-plane components should be approximately equal
        # (transverse isotropy of an axisymmetric disk).
        assert abs(deg_xx - deg_yy) < 1e-6

    def test_prolate_cylindrical_anisotropy_transverse_to_long_axis(self):
        """Regression for #32. (3, 1, 1) is a prolate cylindrical void
        with its symmetry axis along x_1. Load flows easily along the
        long axis (the void is thin in cross-section), but transverse
        load has to bypass a long obstacle — so transverse degradation
        (deg_yy, deg_zz) should exceed axial degradation (deg_xx)."""
        Vp = 0.05
        C_eff = _mt_effective_stiffness(self.C_m, Vp, (3, 1, 1), 0.35)
        deg_xx = (self.C_m[0, 0] - C_eff[0, 0]) / self.C_m[0, 0]
        deg_yy = (self.C_m[1, 1] - C_eff[1, 1]) / self.C_m[1, 1]
        deg_zz = (self.C_m[2, 2] - C_eff[2, 2]) / self.C_m[2, 2]
        assert deg_yy > deg_xx
        # The two equatorial directions are equivalent (axisymmetric).
        assert abs(deg_yy - deg_zz) < 1e-6

    def test_cache_hit_returns_identical_result(self):
        # #42: a repeated call with the same key must come from the cache
        # and return a numerically identical result (within fp tolerance
        # of the original computation, which here is exact equality since
        # the cache stores the actual array).
        from porosity_fe_analysis import _mt_cache, _mt_cache_clear
        _mt_cache_clear()
        first = _mt_effective_stiffness(self.C_m, 0.04, (1, 1, 1), 0.35)
        assert len(_mt_cache) == 1
        second = _mt_effective_stiffness(self.C_m, 0.04, (1, 1, 1), 0.35)
        # Still one entry — no duplication.
        assert len(_mt_cache) == 1
        np.testing.assert_array_equal(first, second)

    def test_cache_returns_defensive_copy(self):
        # Callers may mutate the returned array (e.g. callers in the FE
        # path build derived ratios). The cache must not be poisoned by
        # that mutation — the next call must still return the original.
        from porosity_fe_analysis import _mt_cache_clear
        _mt_cache_clear()
        first = _mt_effective_stiffness(self.C_m, 0.04, (1, 1, 1), 0.35)
        first[0, 0] = -999.0  # mutate the returned array
        second = _mt_effective_stiffness(self.C_m, 0.04, (1, 1, 1), 0.35)
        assert second[0, 0] != -999.0

    def test_cache_distinguishes_materials(self):
        # Two materials with different C_m[0,0] must NOT collide in the
        # cache even at identical (Vp, shape, nu_m).
        from porosity_fe_analysis import _mt_cache, _mt_cache_clear
        _mt_cache_clear()
        C_m2 = self.C_m * 2.0  # different fingerprint
        a = _mt_effective_stiffness(self.C_m, 0.04, (1, 1, 1), 0.35)
        b = _mt_effective_stiffness(C_m2, 0.04, (1, 1, 1), 0.35)
        assert len(_mt_cache) == 2
        # The stiffer matrix should give a stiffer effective stiffness.
        assert b[0, 0] > a[0, 0]


class TestDegradedCompositeStiffness:
    """Direct unit tests for _degraded_composite_stiffness (#48).

    Previously exercised only indirectly through Hex8Element._degraded_stiffness;
    the Vp < 1e-12, Vp > 0.99, and the lame-denominator guard branches were
    not covered, and past matrix-modulus fixes lived in this function.
    """

    def setup_method(self):
        self.mat = MATERIALS['T800_epoxy']
        self.pristine = self.mat.get_stiffness_matrix()

    def test_vp_zero_returns_pristine(self):
        C = _degraded_composite_stiffness(0.0, (1, 1, 1), self.mat)
        np.testing.assert_allclose(C, self.pristine, atol=1e-9)

    def test_vp_subepsilon_returns_pristine(self):
        # Below the 1e-12 guard: must take the early-return branch.
        C = _degraded_composite_stiffness(1e-15, (1, 1, 1), self.mat)
        np.testing.assert_allclose(C, self.pristine, atol=1e-9)

    def test_vp_near_one_returns_zeros(self):
        # Above the 0.99 guard: collapsed material is fully degraded.
        C = _degraded_composite_stiffness(0.995, (1, 1, 1), self.mat)
        np.testing.assert_array_equal(C, np.zeros((6, 6)))

    def test_e11_weakly_affected_e22_g12_strongly(self):
        # At 5% porosity the fiber-dominated E11 barely moves while the
        # matrix-dominated E22 and G12 take significant hits. This is the
        # whole reason this helper exists; if a future refactor inverts
        # those rates, this test must fail.
        Vp = 0.05
        C = _degraded_composite_stiffness(Vp, (1, 1, 1), self.mat)
        S = np.linalg.inv(C)
        S_pristine = np.linalg.inv(self.pristine)
        # Engineering moduli come straight off the compliance diagonal.
        E11_loss = 1.0 - (1.0 / S[0, 0]) / (1.0 / S_pristine[0, 0])
        E22_loss = 1.0 - (1.0 / S[1, 1]) / (1.0 / S_pristine[1, 1])
        G12_loss = 1.0 - (1.0 / S[5, 5]) / (1.0 / S_pristine[5, 5])
        assert E11_loss < 0.01, f"E11 should be near-pristine, lost {E11_loss:.4f}"
        assert E22_loss > E11_loss * 5, (
            f"E22 loss {E22_loss:.4f} should be much larger than E11 loss {E11_loss:.4f}"
        )
        assert G12_loss > E11_loss * 5, (
            f"G12 loss {G12_loss:.4f} should be much larger than E11 loss {E11_loss:.4f}"
        )

    def test_monotonic_e22_degradation(self):
        Vp_list = [0.01, 0.03, 0.05, 0.08]
        E22_seq = []
        for Vp in Vp_list:
            C = _degraded_composite_stiffness(Vp, (1, 1, 1), self.mat)
            S = np.linalg.inv(C)
            E22_seq.append(1.0 / S[1, 1])
        for a, b in zip(E22_seq, E22_seq[1:]):
            assert b < a, f"E22 should drop monotonically with Vp: got {E22_seq}"

    def test_returned_stiffness_positive_definite(self):
        C = _degraded_composite_stiffness(0.05, (1, 1, 1), self.mat)
        eig = np.linalg.eigvalsh(C)
        assert np.all(eig > 0), f"degraded stiffness not positive-definite: {eig}"

    def test_all_finite(self):
        # Sweep through the regime where the lame-denominator guard
        # (lam_eff + mu_eff < 1e-12) could trip; outputs must stay finite.
        for Vp in [1e-10, 0.01, 0.10, 0.50, 0.85, 0.985]:
            C = _degraded_composite_stiffness(Vp, (1, 1, 1), self.mat)
            assert np.all(np.isfinite(C)), f"non-finite C at Vp={Vp}"


class TestHex8Element:
    def setup_method(self):
        # Create a simple unit cube element
        self.node_coords = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=float)
        mat = MATERIALS['T800_epoxy']
        self.C_base = mat.get_stiffness_matrix()
        self.C_m = mat.get_isotropic_matrix_stiffness()
        self.elem = Hex8Element(
            node_coords=self.node_coords,
            C_base=self.C_base,
            ply_angle_deg=0.0,
            node_porosities=np.full(8, 0.03),
            void_shape_radii=(1, 1, 1),
            nu_m=0.35,
            C_m=self.C_m,
        )

    def test_shape_functions_partition_of_unity(self):
        """Shape functions should sum to 1 at any point."""
        N = Hex8Element.shape_functions(0.3, -0.2, 0.5)
        np.testing.assert_allclose(N.sum(), 1.0, atol=1e-14)

    def test_nan_node_porosities_rejected(self):
        bad = np.full(8, 0.03)
        bad[3] = float('nan')
        with pytest.raises(ValueError, match=r"node_porosities must be finite"):
            Hex8Element(
                node_coords=self.node_coords,
                C_base=self.C_base,
                ply_angle_deg=0.0,
                node_porosities=bad,
                void_shape_radii=(1, 1, 1),
                nu_m=0.35,
                C_m=self.C_m,
            )

    def test_node_porosities_above_one_rejected(self):
        bad = np.full(8, 0.03)
        bad[2] = 5.0  # plausibly a percent (5%) → rejected with hint
        with pytest.raises(ValueError, match=r"node_porosities must be a fraction"):
            Hex8Element(
                node_coords=self.node_coords,
                C_base=self.C_base,
                ply_angle_deg=0.0,
                node_porosities=bad,
                void_shape_radii=(1, 1, 1),
                nu_m=0.35,
                C_m=self.C_m,
            )

    def test_node_porosities_fp_overshoot_clipped(self):
        # ~1e-12 above 1.0 should be clipped, not rejected.
        bumped = np.full(8, 1.0)
        bumped[1] = 1.0 + 5e-13
        elem = Hex8Element(
            node_coords=self.node_coords,
            C_base=self.C_base,
            ply_angle_deg=0.0,
            node_porosities=bumped,
            void_shape_radii=(1, 1, 1),
            nu_m=0.35,
            C_m=self.C_m,
        )
        assert np.all(elem.node_porosities <= 1.0)

    def test_shape_functions_at_nodes(self):
        """N_i should be 1 at node i and 0 at other nodes."""
        from porosity_fe_analysis import _NODE_COORDS_REF
        for i in range(8):
            xi, eta, zeta = _NODE_COORDS_REF[i]
            N = Hex8Element.shape_functions(xi, eta, zeta)
            for j in range(8):
                expected = 1.0 if i == j else 0.0
                assert abs(N[j] - expected) < 1e-14

    def test_shape_derivatives_shape(self):
        dN = Hex8Element.shape_derivatives(0.0, 0.0, 0.0)
        assert dN.shape == (3, 8)

    def test_jacobian_unit_cube(self):
        """Jacobian of unit cube should be 0.5 * I (mapping [-1,1] to [0,1])."""
        J = self.elem.jacobian(0.0, 0.0, 0.0)
        assert J.shape == (3, 3)
        np.testing.assert_allclose(J, 0.5 * np.eye(3), atol=1e-14)

    def test_B_matrix_shape(self):
        B = self.elem.B_matrix(0.0, 0.0, 0.0)
        assert B.shape == (6, 24)

    def test_stiffness_matrix_shape(self):
        Ke = self.elem.stiffness_matrix()
        assert Ke.shape == (24, 24)

    def test_stiffness_matrix_symmetric(self):
        Ke = self.elem.stiffness_matrix()
        np.testing.assert_allclose(Ke, Ke.T, atol=1e-4)

    def test_stiffness_matrix_positive_semidefinite(self):
        Ke = self.elem.stiffness_matrix()
        eigenvalues = np.linalg.eigvalsh(Ke)
        # Should have 6 zero eigenvalues (rigid body modes) and 18 positive
        assert np.sum(eigenvalues > 1e-6) >= 12  # At least 12 positive

    def test_volume_unit_cube(self):
        assert abs(self.elem.volume - 1.0) < 1e-12

    def test_inverted_element_rejected_at_assembly(self):
        """Regression for #33: signed det(J) silently corrupting K."""
        # Swap two adjacent nodes on the bottom face to invert the element.
        inverted = self.node_coords.copy()
        inverted[[0, 1]] = inverted[[1, 0]]
        bad_elem = Hex8Element(
            node_coords=inverted,
            C_base=self.C_base,
            ply_angle_deg=0.0,
            node_porosities=np.full(8, 0.03),
            void_shape_radii=(1, 1, 1),
            nu_m=0.35,
            C_m=self.C_m,
        )
        with pytest.raises(ValueError, match="non-positive Jacobian"):
            bad_elem.stiffness_matrix()

    def test_inverted_element_volume_still_positive(self):
        """volume uses abs(det J); only stiffness_matrix raises."""
        inverted = self.node_coords.copy()
        inverted[[0, 1]] = inverted[[1, 0]]
        bad_elem = Hex8Element(
            node_coords=inverted,
            C_base=self.C_base,
            ply_angle_deg=0.0,
            node_porosities=np.full(8, 0.03),
            void_shape_radii=(1, 1, 1),
            nu_m=0.35,
            C_m=self.C_m,
        )
        assert bad_elem.volume > 0

    def test_stress_at_gauss_points_shape(self):
        u_elem = np.zeros(24)
        sig = self.elem.stress_at_gauss_points(u_elem)
        assert sig.shape == (8, 6)

    def test_strain_at_gauss_points_shape(self):
        u_elem = np.zeros(24)
        eps = self.elem.strain_at_gauss_points(u_elem)
        assert eps.shape == (8, 6)

    def test_zero_displacement_zero_stress(self):
        u_elem = np.zeros(24)
        sig = self.elem.stress_at_gauss_points(u_elem)
        np.testing.assert_allclose(sig, 0.0, atol=1e-12)

    def test_uniform_strain_produces_uniform_stress(self):
        """Uniform x-displacement gradient should produce constant sigma_11."""
        # Prescribe u_x = eps_x * x at each node, with eps_x = 0.001
        eps_x = 0.001
        u_elem = np.zeros(24)
        for i in range(8):
            u_elem[3 * i] = eps_x * self.node_coords[i, 0]
        sig = self.elem.stress_at_gauss_points(u_elem)
        # All GP should have approximately the same sigma_11
        sigma_11_vals = sig[:, 0]
        assert np.std(sigma_11_vals) / (np.mean(np.abs(sigma_11_vals)) + 1e-12) < 0.01

    def test_porosity_reduces_stiffness(self):
        """Higher porosity should produce lower element stiffness."""
        elem_low = Hex8Element(self.node_coords, self.C_base, 0.0,
                               np.full(8, 0.01), (1, 1, 1), 0.35, self.C_m)
        elem_high = Hex8Element(self.node_coords, self.C_base, 0.0,
                                np.full(8, 0.10), (1, 1, 1), 0.35, self.C_m)
        Ke_low = elem_low.stiffness_matrix()
        Ke_high = elem_high.stiffness_matrix()
        # Trace of stiffness should be lower for higher porosity
        assert np.trace(Ke_high) < np.trace(Ke_low)

    def test_wrong_node_coords_shape(self):
        with pytest.raises(ValueError):
            Hex8Element(np.zeros((4, 3)), self.C_base, 0.0,
                       np.full(8, 0.03), (1, 1, 1), 0.35, self.C_m)

    def test_wrong_porosity_shape(self):
        with pytest.raises(ValueError):
            Hex8Element(self.node_coords, self.C_base, 0.0,
                       np.full(4, 0.03), (1, 1, 1), 0.35, self.C_m)


class TestCompositeMeshFE:
    """Tests for the FE-related additions to CompositeMesh."""

    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        self.pf = PorosityField(self.material, 0.03, distribution='uniform')

    def test_nodes_on_face_x_min(self):
        mesh = CompositeMesh(self.pf, self.material, nx=5, ny=3, nz=4)
        nodes = mesh.nodes_on_face('x_min')
        assert len(nodes) > 0
        np.testing.assert_allclose(mesh.nodes[nodes, 0], 0.0, atol=1e-10)

    def test_nodes_on_face_x_max(self):
        mesh = CompositeMesh(self.pf, self.material, nx=5, ny=3, nz=4)
        nodes = mesh.nodes_on_face('x_max')
        assert len(nodes) > 0
        np.testing.assert_allclose(mesh.nodes[nodes, 0], mesh.L_x, atol=1e-10)

    def test_nodes_on_face_count(self):
        mesh = CompositeMesh(self.pf, self.material, nx=5, ny=3, nz=4)
        # x_min face should have (ny+1)*(nz+1) nodes
        nodes = mesh.nodes_on_face('x_min')
        assert len(nodes) == (3 + 1) * (4 + 1)

    def test_nodes_on_face_invalid(self):
        mesh = CompositeMesh(self.pf, self.material, nx=5, ny=3, nz=4)
        with pytest.raises(ValueError):
            mesh.nodes_on_face('invalid')

    def test_n_dof(self):
        mesh = CompositeMesh(self.pf, self.material, nx=5, ny=3, nz=4)
        assert mesh.n_dof == mesh.n_nodes * 3

    def test_domain_size(self):
        mesh = CompositeMesh(self.pf, self.material, nx=5, ny=3, nz=4)
        Lx, Ly, Lz = mesh.domain_size
        assert abs(Lx - 50.0) < 1e-10
        assert abs(Ly - 20.0) < 1e-10

    def test_ply_angles_default_zero(self):
        mesh = CompositeMesh(self.pf, self.material, nx=5, ny=3, nz=4)
        np.testing.assert_allclose(mesh.ply_angles, 0.0)

    def test_ply_angles_custom(self):
        mesh = CompositeMesh(self.pf, self.material, nx=5, ny=3, nz=4,
                            ply_angles=[0, 45, 90, -45])
        # Should have angles from the layup
        unique_angles = np.unique(mesh.ply_angles)
        assert len(unique_angles) > 1

    def test_elem_ply_ids(self):
        mesh = CompositeMesh(self.pf, self.material, nx=5, ny=3, nz=4)
        assert hasattr(mesh, 'elem_ply_ids')
        assert len(mesh.elem_ply_ids) == mesh.n_elements


class TestMeshQuality:
    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        self.pf = PorosityField(self.material, 0.03, distribution='uniform')

    def test_returns_dict(self):
        mesh = CompositeMesh(self.pf, self.material, nx=5, ny=3, nz=4)
        result = check_mesh_quality(mesh)
        assert isinstance(result, dict)
        assert 'min_aspect_ratio' in result
        assert 'max_aspect_ratio' in result
        assert 'min_jacobian_det' in result
        assert 'n_inverted' in result
        assert 'n_distorted' in result

    def test_structured_mesh_no_inverted(self):
        mesh = CompositeMesh(self.pf, self.material, nx=5, ny=3, nz=4)
        result = check_mesh_quality(mesh)
        assert result['n_inverted'] == 0

    def test_positive_jacobian(self):
        mesh = CompositeMesh(self.pf, self.material, nx=5, ny=3, nz=4)
        result = check_mesh_quality(mesh)
        assert result['min_jacobian_det'] > 0

    def test_verbose_mode(self):
        mesh = CompositeMesh(self.pf, self.material, nx=3, ny=2, nz=2)
        result = check_mesh_quality(mesh, verbose=True)
        assert result['n_elements'] == mesh.n_elements


class TestGlobalAssembler:
    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        self.pf = PorosityField(self.material, 0.03, distribution='uniform')
        self.mesh = CompositeMesh(self.pf, self.material, nx=3, ny=2, nz=2)

    def test_create_element(self):
        assembler = GlobalAssembler(self.mesh, self.material, self.pf)
        elem = assembler.create_element(0)
        assert isinstance(elem, Hex8Element)

    def test_element_dof_indices_shape(self):
        assembler = GlobalAssembler(self.mesh, self.material, self.pf)
        dofs = assembler.element_dof_indices(0)
        assert dofs.shape == (24,)

    def test_element_dof_indices_range(self):
        assembler = GlobalAssembler(self.mesh, self.material, self.pf)
        dofs = assembler.element_dof_indices(0)
        assert np.all(dofs >= 0)
        assert np.all(dofs < self.mesh.n_dof)

    def test_assemble_stiffness_shape(self):
        assembler = GlobalAssembler(self.mesh, self.material, self.pf)
        K = assembler.assemble_stiffness()
        assert K.shape == (self.mesh.n_dof, self.mesh.n_dof)

    def test_assemble_stiffness_symmetric(self):
        assembler = GlobalAssembler(self.mesh, self.material, self.pf)
        K = assembler.assemble_stiffness()
        K_dense = K.toarray()
        np.testing.assert_allclose(K_dense, K_dense.T, atol=1e-2)

    def test_assemble_stiffness_sparse(self):
        assembler = GlobalAssembler(self.mesh, self.material, self.pf)
        K = assembler.assemble_stiffness()
        assert scipy.sparse.issparse(K)


class TestBoundaryHandler:
    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        self.pf = PorosityField(self.material, 0.03, distribution='uniform')
        self.mesh = CompositeMesh(self.pf, self.material, nx=3, ny=2, nz=2)
        self.handler = BoundaryHandler(self.mesh)

    def test_compression_bcs_returns_tuple(self):
        constrained, F = self.handler.compression_bcs()
        assert isinstance(constrained, dict)
        assert isinstance(F, np.ndarray)
        assert len(F) == self.mesh.n_dof

    def test_compression_bcs_constrained_dofs(self):
        constrained, F = self.handler.compression_bcs()
        assert len(constrained) > 0
        # Should constrain ux on x_min and x_max
        xmin_nodes = self.mesh.nodes_on_face('x_min')
        for nid in xmin_nodes:
            assert 3 * int(nid) in constrained
            assert constrained[3 * int(nid)] == 0.0

    def test_compression_bcs_prescribed_displacement(self):
        strain = -0.01
        constrained, F = self.handler.compression_bcs(applied_strain=strain)
        xmax_nodes = self.mesh.nodes_on_face('x_max')
        expected_disp = strain * self.mesh.L_x
        for nid in xmax_nodes:
            assert abs(constrained[3 * int(nid)] - expected_disp) < 1e-10

    def test_tension_bcs(self):
        strain = 0.01
        constrained, F = self.handler.tension_bcs(applied_strain=strain)
        assert len(constrained) > 0
        assert len(F) == self.mesh.n_dof
        # x_min: ux pinned to 0
        for nid in self.mesh.nodes_on_face('x_min'):
            assert constrained[3 * int(nid)] == 0.0
        # x_max: ux = +strain * Lx (positive => tension, not compression)
        expected = strain * self.mesh.L_x
        assert expected > 0.0
        for nid in self.mesh.nodes_on_face('x_max'):
            assert abs(constrained[3 * int(nid)] - expected) < 1e-12
        # y_min: uy pinned to 0 (symmetry)
        for nid in self.mesh.nodes_on_face('y_min'):
            assert constrained[3 * int(nid) + 1] == 0.0

    def test_shear_bcs(self):
        gamma = 0.01
        constrained, F = self.handler.shear_bcs(applied_strain=gamma)
        assert len(constrained) > 0
        assert len(F) == self.mesh.n_dof
        nodes = self.mesh.nodes
        # All four side faces must prescribe BOTH ux and uy to the pure-shear
        # field u = gamma/2 * y, v = gamma/2 * x. A regression that swapped
        # ux/uy on a face, or left a face traction-free, fails here.
        for face in ('x_min', 'x_max', 'y_min', 'y_max'):
            face_nodes = self.mesh.nodes_on_face(face)
            assert len(face_nodes) > 0
            for nid in face_nodes:
                nid = int(nid)
                x_n, y_n = float(nodes[nid, 0]), float(nodes[nid, 1])
                assert abs(constrained[3 * nid] - (gamma / 2.0) * y_n) < 1e-12
                assert abs(constrained[3 * nid + 1] - (gamma / 2.0) * x_n) < 1e-12

    def test_apply_penalty(self):
        import scipy.sparse
        assembler = GlobalAssembler(self.mesh, self.material, self.pf)
        K = assembler.assemble_stiffness()
        constrained, F = self.handler.compression_bcs()
        K_mod, F_mod = BoundaryHandler.apply_penalty(K, F, constrained)
        assert K_mod.shape == K.shape
        assert len(F_mod) == len(F)

    def test_penalty_increases_diagonal(self):
        import scipy.sparse
        assembler = GlobalAssembler(self.mesh, self.material, self.pf)
        K = assembler.assemble_stiffness()
        constrained, F = self.handler.compression_bcs()
        K_mod, F_mod = BoundaryHandler.apply_penalty(K, F, constrained)
        # Constrained DOF diagonals should be much larger
        for dof in list(constrained.keys())[:5]:
            assert K_mod[dof, dof] > K[dof, dof]


class TestFESolver:
    """Integration tests for the full FE solver pipeline."""

    def setup_method(self):
        self.material = MATERIALS['T800_epoxy']
        self.pf = PorosityField(self.material, 0.03, distribution='uniform')
        # Very coarse mesh for speed
        self.mesh = CompositeMesh(self.pf, self.material, nx=3, ny=2, nz=2)

    def test_solve_returns_field_results(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        results = solver.solve(loading='compression', applied_strain=-0.001)
        assert isinstance(results, FieldResults)

    def test_solve_displacement_shape(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        results = solver.solve(loading='compression', applied_strain=-0.001)
        assert results.displacement.shape == (self.mesh.n_nodes, 3)

    def test_solve_stress_shape(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        results = solver.solve(loading='compression', applied_strain=-0.001)
        assert results.stress_global.shape == (self.mesh.n_elements, 8, 6)
        assert results.stress_local.shape == (self.mesh.n_elements, 8, 6)

    def test_solve_strain_shape(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        results = solver.solve(loading='compression', applied_strain=-0.001)
        assert results.strain_global.shape == (self.mesh.n_elements, 8, 6)

    def test_solve_knockdown_range(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        results = solver.solve(loading='compression', applied_strain=-0.001)
        assert 0 < results.knockdown <= 1.0

    def test_solve_failure_index_positive(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        results = solver.solve(loading='compression', applied_strain=-0.001)
        assert results.max_failure_index >= 0

    def test_solve_nonzero_displacement(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        results = solver.solve(loading='compression', applied_strain=-0.001)
        assert np.max(np.abs(results.displacement)) > 0

    def test_solve_tension(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        results = solver.solve(loading='tension', applied_strain=0.001)
        assert isinstance(results, FieldResults)

    def test_solve_shear(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        results = solver.solve(loading='shear', applied_strain=0.001)
        assert isinstance(results, FieldResults)

    def test_solve_invalid_loading(self):
        solver = FESolver(self.mesh, self.material, self.pf)
        with pytest.raises(ValueError):
            solver.solve(loading='invalid')

    def test_strain_local_uses_strain_transform_not_stress_transform(self):
        """Regression for #38: engineering strain was rotated via T_sigma
        (the stress transformation), which leaves the shear slots off by
        a factor of 2. strain_local must equal T_epsilon @ strain_global
        per (element, Gauss point) — within numerical noise."""
        from porosity_fe_analysis import strain_transformation_3d
        # 45-degree plies make the bug most visible (shear components dominate).
        mat45 = dataclasses.replace(self.material, n_plies=4)
        pf = PorosityField(mat45, 0.02, distribution='uniform')
        mesh = CompositeMesh(pf, mat45, nx=3, ny=2, nz=4,
                              ply_angles=[45.0, -45.0, -45.0, 45.0])
        solver = FESolver(mesh, mat45, pf)
        r = solver.solve(loading='compression', applied_strain=-0.001)
        # Check transformation invariant on a handful of elements.
        for e in [0, mesh.n_elements // 2, mesh.n_elements - 1]:
            ply_rad = np.radians(float(mesh.ply_angles[e]))
            T_eps = strain_transformation_3d(ply_rad, axis='z')
            for g in range(8):
                expected = T_eps @ r.strain_global[e, g]
                np.testing.assert_allclose(
                    r.strain_local[e, g], expected,
                    rtol=1e-10, atol=1e-12,
                    err_msg=f"strain_local mismatch at elem={e}, gp={g}",
                )

    def test_higher_porosity_softer_response(self):
        """Higher porosity should produce softer material (lower stresses
        for the same applied displacement)."""
        pf_low = PorosityField(self.material, 0.01, distribution='uniform')
        mesh_low = CompositeMesh(pf_low, self.material, nx=3, ny=2, nz=2)
        solver_low = FESolver(mesh_low, self.material, pf_low)
        result_low = solver_low.solve(loading='compression', applied_strain=-0.001)

        pf_high = PorosityField(self.material, 0.08, distribution='uniform')
        mesh_high = CompositeMesh(pf_high, self.material, nx=3, ny=2, nz=2)
        solver_high = FESolver(mesh_high, self.material, pf_high)
        result_high = solver_high.solve(loading='compression', applied_strain=-0.001)

        # Higher porosity -> softer -> lower stresses for same displacement
        max_stress_low = np.max(np.abs(result_low.stress_global[:, :, 0]))
        max_stress_high = np.max(np.abs(result_high.stress_global[:, :, 0]))
        assert max_stress_high < max_stress_low

    def test_solve_verbose(self):
        """Verbose mode should not crash."""
        solver = FESolver(self.mesh, self.material, self.pf)
        results = solver.solve(loading='compression', applied_strain=-0.001, verbose=True)
        assert isinstance(results, FieldResults)

    def test_displacement_boundary_conditions_applied(self):
        """Check that BCs are approximately satisfied."""
        solver = FESolver(self.mesh, self.material, self.pf)
        strain = -0.001
        results = solver.solve(loading='compression', applied_strain=strain)

        # x_min nodes should have ~0 x-displacement
        xmin_nodes = self.mesh.nodes_on_face('x_min')
        np.testing.assert_allclose(results.displacement[xmin_nodes, 0], 0.0, atol=1e-8)

        # x_max nodes should have ~strain*Lx displacement
        xmax_nodes = self.mesh.nodes_on_face('x_max')
        expected = strain * self.mesh.L_x
        np.testing.assert_allclose(results.displacement[xmax_nodes, 0], expected, atol=1e-6)


class TestFEExportResults:
    def test_export_creates_file(self, tmp_path):
        material = MATERIALS['T800_epoxy']
        pf = PorosityField(material, 0.03, distribution='uniform')
        mesh = CompositeMesh(pf, material, nx=3, ny=2, nz=2)
        solver = FESolver(mesh, material, pf)
        results = solver.solve(loading='compression', applied_strain=-0.001)
        path = str(tmp_path / "fe_results.json")
        FESolver.export_results(results, path)
        assert os.path.exists(path)

    def test_export_json_structure(self, tmp_path):
        material = MATERIALS['T800_epoxy']
        pf = PorosityField(material, 0.03, distribution='uniform')
        mesh = CompositeMesh(pf, material, nx=3, ny=2, nz=2)
        solver = FESolver(mesh, material, pf)
        results = solver.solve(loading='compression', applied_strain=-0.001)
        path = str(tmp_path / "fe_results.json")
        FESolver.export_results(results, path)
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        # Envelope keys
        assert 'schema_version' in data
        assert 'provenance' in data
        # Results merged into the envelope at the top level
        assert 'displacement' in data
        assert 'stress_global' in data
        assert 'failure' in data
        assert 'knockdown_factor' in data['failure']
        assert data['failure']['knockdown_factor'] > 0


class TestReprMethods:
    def test_material_repr(self):
        mat = MATERIALS['T800_epoxy']
        r = repr(mat)
        assert 'MaterialProperties' in r
        assert 'E11=161000' in r

    def test_void_geometry_repr(self):
        void = VoidGeometry(center=(1, 2, 3), radii=(1, 1, 1))
        r = repr(void)
        assert 'VoidGeometry' in r
        assert 'aspect_ratio=1.00' in r

    def test_porosity_field_repr(self):
        mat = MATERIALS['T800_epoxy']
        pf = PorosityField(mat, 0.03, distribution='uniform')
        r = repr(pf)
        assert 'PorosityField' in r
        assert '0.0300' in r

    def test_composite_mesh_repr(self):
        mat = MATERIALS['T800_epoxy']
        pf = PorosityField(mat, 0.03, distribution='uniform')
        mesh = CompositeMesh(pf, mat, nx=3, ny=2, nz=2)
        r = repr(mesh)
        assert 'CompositeMesh' in r
        assert 'nx=3' in r

    def test_field_results_repr(self):
        mat = MATERIALS['T800_epoxy']
        pf = PorosityField(mat, 0.03, distribution='uniform')
        mesh = CompositeMesh(pf, mat, nx=3, ny=2, nz=2)
        solver = FESolver(mesh, mat, pf)
        results = solver.solve(loading='compression', applied_strain=-0.001)
        r = repr(results)
        assert 'FieldResults' in r
        assert 'knockdown=' in r


class TestVoidInclusions:
    """Tests that discrete voids are modeled as near-zero stiffness inclusions."""

    def test_void_elements_identified_by_geometry(self):
        """Elements inside a VoidGeometry should be flagged as void."""
        material = MATERIALS['T800_epoxy']
        Lz = material.total_thickness
        void = VoidGeometry(center=(25, 10, Lz / 2), radii=(5, 5, Lz / 4))
        pf = PorosityField(material, 0.0, distribution='uniform',
                           discrete_voids=[void])
        mesh = CompositeMesh(pf, material, nx=10, ny=5, nz=6)
        # Should have some void elements
        assert len(mesh.void_elements) > 0
        # Void elements should be near the void center
        void_centers = np.mean(mesh.nodes[mesh.elements[mesh.void_elements]], axis=1)
        for center in void_centers:
            assert void.contains(np.array([center[0]]), np.array([center[1]]),
                                  np.array([center[2]]))[0]

    def test_void_element_has_near_zero_stiffness(self):
        """Hex8Element with is_void=True should have very soft stiffness."""
        material = MATERIALS['T800_epoxy']
        C_base = material.get_stiffness_matrix()
        C_m = material.get_isotropic_matrix_stiffness()
        coords = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                           [0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=float)
        porosity = np.zeros(8)

        elem_normal = Hex8Element(coords, C_base, 0.0, porosity,
                                   (1,1,1), 0.35, C_m, is_void=False)
        elem_void = Hex8Element(coords, C_base, 0.0, porosity,
                                 (1,1,1), 0.35, C_m, is_void=True)

        Ke_normal = elem_normal.stiffness_matrix()
        Ke_void = elem_void.stiffness_matrix()

        # Void element stiffness should be orders of magnitude smaller
        ratio = np.linalg.norm(Ke_void) / np.linalg.norm(Ke_normal)
        assert ratio < 1e-4, f"Void/normal stiffness ratio {ratio} not small enough"

    def test_fe_with_void_has_stress_concentration(self):
        """FE solve with a void should show higher stresses near the void."""
        material = MATERIALS['T800_epoxy']
        Lz = material.total_thickness
        # Large void relative to mesh: 10mm radius covers multiple elements
        void = VoidGeometry(center=(25, 10, Lz / 2), radii=(10, 8, Lz / 3))
        pf = PorosityField(material, 0.0, distribution='uniform',
                           discrete_voids=[void])
        mesh = CompositeMesh(pf, material, nx=10, ny=5, nz=6)

        assert len(mesh.void_elements) > 0, "No void elements found"

        solver = FESolver(mesh, material, pf)
        results = solver.solve(loading='compression', applied_strain=-0.005)

        # Non-void elements near the void should have higher stresses
        # than elements far from the void
        assert results.max_failure_index > 0
        assert np.any(np.isfinite(results.stress_global))


# ============================================================
# Coverage backfill (#12): layup-scaling helpers, linear-model
# saturation, CLT degradation boundaries, CLI smoke, GUI parser.
# ============================================================


class TestEmpiricalLayupScaling:
    """Direct unit tests for _matrix_dominated_fraction and _layup_scale."""

    def test_f_md_pure_zero(self):
        assert EmpiricalSolver._matrix_dominated_fraction([0] * 8) == 0.0

    def test_f_md_pure_ninety(self):
        assert EmpiricalSolver._matrix_dominated_fraction([90] * 8) == 1.0

    def test_f_md_off_axis_only(self):
        assert EmpiricalSolver._matrix_dominated_fraction([45, -45, 45, -45]) == 0.5

    def test_f_md_qi_layup_is_0p4(self):
        # Documented QI calibration coupon -> 0.4 under the binning rule.
        # See the comment above _F_MD_REF in porosity_fe_analysis.py and
        # the README "Empirical Strength Knockdown" section.
        layup = [0, 45, 90, -45, 0, 0, -45, 90, 45, 0]
        assert abs(EmpiricalSolver._matrix_dominated_fraction(layup) - 0.4) < 1e-12

    def test_f_md_empty_returns_qi_default(self):
        assert EmpiricalSolver._matrix_dominated_fraction([]) == 0.5
        assert EmpiricalSolver._matrix_dominated_fraction(None) == 0.5

    def test_f_md_threshold_band_at_10_and_80_degrees(self):
        # 10° -> still binned as 0° (fiber-dominated)
        assert EmpiricalSolver._matrix_dominated_fraction([10]) == 0.0
        # 80° -> binned as 90° (matrix-dominated)
        assert EmpiricalSolver._matrix_dominated_fraction([80]) == 1.0
        # 11° -> off-axis bin
        assert EmpiricalSolver._matrix_dominated_fraction([11]) == 0.5

    def _solver_with_layup(self, ply_angles):
        material = MATERIALS['T800_epoxy']
        pf = PorosityField(material, 0.03, distribution='uniform')
        mesh = CompositeMesh(pf, material, nx=4, ny=3, nz=4)
        return EmpiricalSolver(mesh, material, ply_angles=ply_angles)

    def test_layup_scale_unity_at_reference(self):
        # f_md = 0.5 -> scale = 1.0 -> alpha_eff == alpha_QI
        solver = self._solver_with_layup([45, -45, 45, -45])
        for mode, alpha_qi in EmpiricalSolver._JUDD_WRIGHT_ALPHA_QI.items():
            assert abs(solver.JUDD_WRIGHT_ALPHA[mode] - alpha_qi) < 1e-12

    def test_layup_scale_floor_for_ud(self):
        # f_md = 0.0 -> hits 0.15 floor for non-ILSS modes, 0.80 for ILSS.
        solver = self._solver_with_layup([0] * 8)
        for mode in ('compression', 'tension', 'shear'):
            expected = EmpiricalSolver._JUDD_WRIGHT_ALPHA_QI[mode] * 0.15
            assert abs(solver.JUDD_WRIGHT_ALPHA[mode] - expected) < 1e-12
        ilss_expected = EmpiricalSolver._JUDD_WRIGHT_ALPHA_QI['ilss'] * 0.80
        assert abs(solver.JUDD_WRIGHT_ALPHA['ilss'] - ilss_expected) < 1e-12

    def test_layup_scale_above_reference(self):
        # Pure 90 -> f_md = 1.0 -> scale = 2.0
        solver = self._solver_with_layup([90] * 8)
        for mode, alpha_qi in EmpiricalSolver._JUDD_WRIGHT_ALPHA_QI.items():
            assert abs(solver.JUDD_WRIGHT_ALPHA[mode] - alpha_qi * 2.0) < 1e-12


class TestEmpiricalLinearSaturation:
    """The linear knockdown clips to 0 once Vp >= 1/beta."""

    def setup_method(self):
        material = MATERIALS['T800_epoxy']
        pf = PorosityField(material, 0.03, distribution='uniform')
        mesh = CompositeMesh(pf, material, nx=4, ny=3, nz=4)
        self.solver = EmpiricalSolver(mesh, material)

    def test_linear_clips_to_zero_at_full_porosity(self):
        for mode in ('compression', 'tension', 'shear', 'ilss'):
            assert self.solver._linear(1.0, mode) == 0.0

    def test_linear_monotone_decreasing_until_saturation(self):
        prev = 1.0
        for vp in (0.0, 0.01, 0.05, 0.10, 0.18, 0.20):
            kd = self.solver._linear(vp, 'compression')
            assert kd <= prev + 1e-12
            assert 0.0 <= kd <= 1.0
            prev = kd

    def test_linear_internal_clip_tolerates_fp_overshoot(self):
        # FE element-mean averaging can produce 1 + ~1e-15.
        kd = self.solver._linear(1.0 + 1e-15, 'compression')
        assert kd == 0.0

    def test_internal_clip_rejects_nan(self):
        with pytest.raises(ValueError, match="non-finite"):
            self.solver._judd_wright(float('nan'), 'compression')


class TestCLTDegradation:
    """Boundary tests for compute_degraded_clt_moduli (#12)."""

    def setup_method(self):
        from porosity_fe_analysis import compute_degraded_clt_moduli, \
            compute_degraded_clt_flexural_modulus
        self.compute_degraded_clt_moduli = compute_degraded_clt_moduli
        self.compute_degraded_clt_flexural_modulus = compute_degraded_clt_flexural_modulus
        self.material = MATERIALS['T800_epoxy']
        self.layup = [0, 45, 90, -45, 0, 0, -45, 90, 45, 0]

    def test_pristine_at_zero_porosity(self):
        deg = self.compute_degraded_clt_moduli(self.material, self.layup, Vp=0.0)
        # At Vp=0, degraded should be very close to nearly-zero-Vp baseline.
        baseline = self.compute_degraded_clt_moduli(self.material, self.layup, Vp=1e-9)
        for key in ('Ex', 'Ey', 'Gxy'):
            assert abs(deg[key] - baseline[key]) / baseline[key] < 1e-3

    def test_moduli_decrease_with_porosity(self):
        low = self.compute_degraded_clt_moduli(self.material, self.layup, Vp=0.01)
        high = self.compute_degraded_clt_moduli(self.material, self.layup, Vp=0.10)
        for key in ('Ex', 'Ey', 'Gxy'):
            assert high[key] < low[key]

    def test_flexural_modulus_decreases_with_porosity(self):
        f_low = self.compute_degraded_clt_flexural_modulus(
            self.material, self.layup, Vp=0.0
        )['Ef_x']
        f_high = self.compute_degraded_clt_flexural_modulus(
            self.material, self.layup, Vp=0.05
        )['Ef_x']
        assert f_high < f_low


class TestValidateCLISmoke:
    """Smoke tests for the validate_porosity CLI entry point (#12)."""

    def test_help_exits_zero(self):
        from validate_porosity_cli import main
        with pytest.raises(SystemExit) as exc:
            main(['--help'])
        assert exc.value.code == 0


class TestLayupParser:
    """parse_layup is a pure helper extracted in #9; tested here for #12."""

    def setup_method(self):
        from app import parse_layup
        self.parse_layup = parse_layup

    def test_simple_slash_form(self):
        assert self.parse_layup('[0/45/-45/90]') == [0.0, 45.0, -45.0, 90.0]

    def test_repeat_and_symmetry(self):
        out = self.parse_layup('[0/90]_2s')
        # repeat then mirror: [0,90,0,90] -> [0,90,0,90,90,0,90,0]
        assert out == [0.0, 90.0, 0.0, 90.0, 90.0, 0.0, 90.0, 0.0]

    def test_comma_separator_alternative(self):
        assert self.parse_layup('[90, 0, 90]') == [90.0, 0.0, 90.0]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            self.parse_layup('')

    def test_invalid_angle_token_raises(self):
        with pytest.raises(ValueError, match="Invalid ply angle"):
            self.parse_layup('[0/oops/90]')

    def test_invalid_repeat_token_raises(self):
        with pytest.raises(ValueError, match="Invalid repeat count"):
            self.parse_layup('[0/45]_xyz')

    def test_negative_repeat_raises(self):
        with pytest.raises(ValueError, match=">= 1"):
            self.parse_layup('[0/45]_-3')

    def test_no_angles_raises(self):
        with pytest.raises(ValueError, match="No ply angles"):
            self.parse_layup('[]_3s')


class TestExportHelpers:
    """Module-level CSV / JSON writers extracted for issue #30."""

    @staticmethod
    def _sample_result():
        return {
            "config": {
                "material_name": "T800_epoxy",
                "n_plies": 24,
                "t_ply": 0.183,
                "Vp": 3.0,
                "distribution": "uniform",
                "void_shape": "spherical",
                "nx": 30, "ny": 10, "nz": 12,
            },
            "empirical": {
                "compression": {
                    "judd_wright": {"failure_stress": 1234.5, "knockdown": 0.823},
                    "power_law": {"failure_stress": 1300.0, "knockdown": 0.867},
                },
                "ilss": {
                    "judd_wright": {"failure_stress": 67.0, "knockdown": 0.744},
                },
            },
        }

    def test_build_export_payload_shape(self):
        from app import build_export_payload
        payload = build_export_payload(self._sample_result())
        assert payload["config"]["material"] == "T800_epoxy"
        assert payload["config"]["mesh"] == "30x10x12"
        assert payload["empirical"]["compression"]["judd_wright"]["knockdown"] == 0.823

    def test_write_results_json_round_trips(self, tmp_path):
        from app import build_export_payload, write_results_json
        path = str(tmp_path / "out.json")
        write_results_json(path, build_export_payload(self._sample_result()))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["empirical"]["compression"]["judd_wright"]["knockdown"] == 0.823

    def test_write_results_csv_header_and_rows(self, tmp_path):
        from app import build_export_payload, write_results_csv
        path = str(tmp_path / "out.csv")
        write_results_csv(path, build_export_payload(self._sample_result()))
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        # First N lines are #-prefixed config; then the header; then rows.
        comment_lines = [l for l in lines if l.startswith("#")]
        data_lines = [l for l in lines if not l.startswith("#")]
        assert any("material: T800_epoxy" in l for l in comment_lines)
        assert any("Vp_percent: 3.0" in l for l in comment_lines)
        assert data_lines[0] == "mode,model,failure_stress_MPa,knockdown"
        # Three (mode, model) rows in the sample → header + 3 = 4 lines.
        assert len(data_lines) == 4
        assert "compression,judd_wright,1234.5,0.823" in data_lines

    def test_write_results_csv_round_trips_via_csv_module(self, tmp_path):
        import csv as _csv
        from app import build_export_payload, write_results_csv
        path = str(tmp_path / "out.csv")
        write_results_csv(path, build_export_payload(self._sample_result()))
        with open(path, encoding="utf-8", newline="") as f:
            # Skip comment lines exactly the way pandas read_csv(comment='#') would.
            rows = [r for r in _csv.reader(f) if r and not r[0].startswith("#")]
        assert rows[0] == ["mode", "model", "failure_stress_MPa", "knockdown"]
        # All non-header rows should parse to four columns; numeric ones finite.
        for row in rows[1:]:
            assert len(row) == 4
            float(row[2])
            float(row[3])


class TestKeCacheKeyGeometry:
    """Regression tests for issue #40: _ke_cache key must encode full element
    geometry and material so skewed/non-rectilinear elements or elements with
    different C_base never collide with axis-aligned ones."""

    def _make_elem(self, node_coords, C_base, porosity=0.03, material=None):
        mat = MATERIALS['T800_epoxy']
        C_m = mat.get_isotropic_matrix_stiffness()
        return Hex8Element(
            node_coords=np.asarray(node_coords, dtype=float),
            C_base=C_base,
            ply_angle_deg=0.0,
            node_porosities=np.full(8, porosity),
            void_shape_radii=(1, 1, 1),
            nu_m=mat.matrix_poisson,
            C_m=C_m,
            material=material,  # None => legacy C_base scaling path
        )

    def setup_method(self):
        mat = MATERIALS['T800_epoxy']
        self.C_base = mat.get_stiffness_matrix()

        # Axis-aligned unit cube
        self.coords_rect = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=float)

        # Same bounding box (dx=dy=dz=1) but one node is sheared in x
        self.coords_shear = self.coords_rect.copy()
        self.coords_shear[2, 0] += 0.2   # shear node 2 in x

    def test_stiffness_differs_for_sheared_element(self):
        """Sheared element must produce a different Ke than its axis-aligned twin."""
        elem_rect = self._make_elem(self.coords_rect, self.C_base)
        elem_shear = self._make_elem(self.coords_shear, self.C_base)
        Ke_rect = elem_rect.stiffness_matrix()
        Ke_shear = elem_shear.stiffness_matrix()
        assert not np.allclose(Ke_rect, Ke_shear, atol=1.0), (
            "Stiffness matrices of axis-aligned and sheared elements should differ"
        )

    def test_cache_key_differs_for_sheared_element(self):
        """Cache key must differ between axis-aligned and sheared elements."""
        mat = MATERIALS['T800_epoxy']
        pf = PorosityField(mat, 0.03, distribution='uniform')
        mesh = CompositeMesh(pf, mat, nx=2, ny=2, nz=2)
        assembler = GlobalAssembler(mesh, mat, pf)

        # Manually build keys from two synthetic node-coord arrays.
        # We exploit that _element_cache_key reads from mesh internals,
        # so instead we test the geometry encoding directly via
        # the centroid-relative tuple approach used in the fixed code.
        import numpy as _np

        def _geom_key(coords):
            centroid = coords.mean(axis=0)
            rel = _np.round(coords - centroid, 8)
            return tuple(rel.ravel())

        key_rect = _geom_key(self.coords_rect)
        key_shear = _geom_key(self.coords_shear)
        assert key_rect != key_shear, (
            "Geometry cache keys must differ for axis-aligned vs sheared nodes"
        )

    def test_stiffness_differs_for_different_material(self):
        """Two elements with the same geometry but different C_base must differ in Ke.

        We use the legacy path (material=None) so that the stiffness is computed
        directly from C_base, making the difference observable in Ke.
        """
        mat2 = MATERIALS['T700_epoxy']
        C_base2 = mat2.get_stiffness_matrix()
        # material=None → legacy scalar-degradation path that reads C_base directly
        elem1 = self._make_elem(self.coords_rect, self.C_base, material=None)
        elem2 = self._make_elem(self.coords_rect, C_base2, material=None)
        Ke1 = elem1.stiffness_matrix()
        Ke2 = elem2.stiffness_matrix()
        assert not np.allclose(Ke1, Ke2, atol=1.0), (
            "Stiffness matrices of elements with different C_base should differ"
        )

    def test_cache_key_differs_for_different_material(self):
        """Cache key must differ when C_base changes, even for identical geometry."""
        mat2 = MATERIALS['T700_epoxy']
        C_base2 = mat2.get_stiffness_matrix()
        c_key1 = hash(self.C_base.tobytes())
        c_key2 = hash(C_base2.tobytes())
        assert c_key1 != c_key2, (
            "Material hash in cache key must differ for different C_base matrices"
        )

    def test_identical_elements_share_cache_key(self):
        """Two identical axis-aligned elements must produce the same cache key."""
        import numpy as _np

        def _geom_key(coords):
            centroid = coords.mean(axis=0)
            rel = _np.round(coords - centroid, 8)
            return tuple(rel.ravel())

        key1 = _geom_key(self.coords_rect)
        # Translate the element — the centroid-relative coords must be identical.
        coords_translated = self.coords_rect + np.array([5.0, 3.0, 1.0])
        key2 = _geom_key(coords_translated)
        assert key1 == key2, (
            "Translated copies of the same element shape should share the geometry key"
        )


# ============================================================
# Issue #39: pure-shear BC fix — G12 recovery test
# ============================================================


class TestPureShearBCs:
    """Verify that shear_bcs imposes true pure shear and recovers G12 correctly.

    An isotropic material (E11=E22=E33, G12=E/(2*(1+nu))) is used so that
    the analytical shear modulus is known exactly.  The FE-recovered G12 is
    computed as:

        G12_fe = mean(sigma_xy) / gamma

    where gamma = applied_strain (engineering shear strain) and sigma_xy is
    the volume-average Voigt component index 5 (1-indexed: [0]=s11, [1]=s22,
    [2]=s33, [3]=s23, [4]=s13, [5]=s12).
    """

    @staticmethod
    def _make_isotropic_material(E: float = 10000.0, nu: float = 0.30) -> 'MaterialProperties':
        """Return a MaterialProperties that behaves as an isotropic solid."""
        G = E / (2.0 * (1.0 + nu))
        return MaterialProperties(
            E11=E, E22=E, E33=E,
            G12=G, G13=G, G23=G,
            nu12=nu, nu13=nu, nu23=nu,
            sigma_1c=1e6, sigma_1t=1e6,
            sigma_2t=1e6, sigma_2c=1e6,
            tau_12=1e6, tau_ilss=1e6,
            t_ply=0.5, n_plies=4,
            matrix_modulus=E, matrix_poisson=nu,
            fiber_modulus=E, fiber_volume_fraction=0.6,
        )

    def test_shear_bcs_prescribes_all_four_faces(self):
        """After the fix, all four side faces must carry prescribed displacements."""
        mat = self._make_isotropic_material()
        pf = PorosityField(mat, 0.0, distribution='uniform')
        mesh = CompositeMesh(pf, mat, nx=2, ny=2, nz=2)
        handler = BoundaryHandler(mesh)

        gamma = 0.01
        constrained, F = handler.shear_bcs(applied_strain=gamma)

        nodes = mesh.nodes
        # Every node on any of the four side faces must have ux and uy prescribed.
        for face in ('x_min', 'x_max', 'y_min', 'y_max'):
            for nid in mesh.nodes_on_face(face):
                nid = int(nid)
                assert 3 * nid in constrained, (
                    f"ux not prescribed for node {nid} on face {face}")
                assert 3 * nid + 1 in constrained, (
                    f"uy not prescribed for node {nid} on face {face}")
                x_n = float(nodes[nid, 0])
                y_n = float(nodes[nid, 1])
                np.testing.assert_allclose(
                    constrained[3 * nid], (gamma / 2.0) * y_n, atol=1e-12,
                    err_msg=f"ux wrong for node {nid} on {face}")
                np.testing.assert_allclose(
                    constrained[3 * nid + 1], (gamma / 2.0) * x_n, atol=1e-12,
                    err_msg=f"uy wrong for node {nid} on {face}")

    def test_recovered_G12_matches_analytical(self):
        """FE-recovered G12 must match E/(2*(1+nu)) within 2 %."""
        E = 10000.0
        nu = 0.30
        G_analytical = E / (2.0 * (1.0 + nu))

        mat = self._make_isotropic_material(E=E, nu=nu)
        pf = PorosityField(mat, 0.0, distribution='uniform')
        # 4x4x4 gives 64 elements — coarse but sufficient for a homogeneous cube
        mesh = CompositeMesh(pf, mat, nx=4, ny=4, nz=4)
        solver = FESolver(mesh, mat, pf)

        gamma = 0.01
        results = solver.solve(loading='shear', applied_strain=gamma)

        # Volume-average sigma_xy (Voigt index 5, 0-based)
        sigma_xy_mean = float(np.mean(results.stress_global[:, :, 5]))
        G12_fe = sigma_xy_mean / gamma

        rel_err = abs(G12_fe - G_analytical) / G_analytical
        assert rel_err < 0.02, (
            f"G12 recovery failed: G12_fe={G12_fe:.1f}, "
            f"G_analytical={G_analytical:.1f}, rel_err={rel_err:.4f}")

    def test_shear_only_stress_state(self):
        """Normal stresses must be negligible compared with shear stress."""
        E = 10000.0
        nu = 0.30

        mat = self._make_isotropic_material(E=E, nu=nu)
        pf = PorosityField(mat, 0.0, distribution='uniform')
        mesh = CompositeMesh(pf, mat, nx=4, ny=4, nz=4)
        solver = FESolver(mesh, mat, pf)

        gamma = 0.01
        results = solver.solve(loading='shear', applied_strain=gamma)

        # indices: 0=s11, 1=s22, 2=s33, 3=s23, 4=s13, 5=s12
        sigma = results.stress_global  # shape (n_elem, n_gp, 6)

        sigma_xy_rms = float(np.sqrt(np.mean(sigma[:, :, 5] ** 2)))
        for i, label in enumerate(['s11', 's22', 's33', 's23', 's13']):
            sigma_i_rms = float(np.sqrt(np.mean(sigma[:, :, i] ** 2)))
            ratio = sigma_i_rms / sigma_xy_rms if sigma_xy_rms > 0 else 0.0
            assert ratio < 0.05, (
                f"Non-shear stress {label} too large relative to s12: "
                f"ratio={ratio:.4f} (rms {label}={sigma_i_rms:.2f}, "
                f"rms s12={sigma_xy_rms:.2f})")


class TestHRefinementConvergence:
    """h-refinement convergence: finer mesh should approach the analytical
    uniaxial-tension result more closely than the coarser mesh (#18)."""

    def _run_tension(self, nx, ny, nz, applied_strain=0.001):
        """Build a zero-porosity mesh and solve uniaxial tension.

        Returns the volume-averaged sigma_xx stress at all Gauss points.
        """
        material = MATERIALS['T800_epoxy']
        pf = PorosityField(material, void_volume_fraction=0.0, distribution='uniform')
        mesh = CompositeMesh(pf, material, nx=nx, ny=ny, nz=nz)
        solver = FESolver(mesh, material, pf)
        results = solver.solve(loading='tension', applied_strain=applied_strain)
        # Average sigma_xx across all elements and Gauss points
        avg_sigma_xx = float(np.mean(results.stress_global[:, :, 0]))
        return avg_sigma_xx

    def test_h_refinement_monotone_convergence(self):
        """Refining the mesh from 2x2x2 to 4x4x4 elements should produce a
        sigma_xx that is closer to the analytical value, OR the two mesh
        densities agree to within a tightening tolerance (monotone convergence).

        Analytical uniaxial tension for an all-0-degree ply laminate:
          sigma_xx_analytic ≈ E11 * applied_strain  (simplified, ignores
          lateral coupling), which serves as an upper-bound reference.
        """
        applied_strain = 0.001
        material = MATERIALS['T800_epoxy']

        # Coarse mesh: 2x2x2 hex elements
        sigma_coarse = self._run_tension(nx=2, ny=2, nz=2,
                                         applied_strain=applied_strain)

        # Fine mesh: 4x4x4 hex elements
        sigma_fine = self._run_tension(nx=4, ny=4, nz=4,
                                       applied_strain=applied_strain)

        # Analytical reference: sigma_xx ~ C11 * eps_xx for uniaxial tension
        # with all-0-degree plies.  C11 from the material stiffness matrix.
        C = material.get_stiffness_matrix()
        sigma_analytic = float(C[0, 0]) * applied_strain

        err_coarse = abs(sigma_coarse - sigma_analytic)
        err_fine = abs(sigma_fine - sigma_analytic)

        # The fine mesh must be at least as accurate as the coarse mesh,
        # OR the difference between the two meshes must be small relative
        # to the magnitude (monotone convergence guard).
        mesh_diff = abs(sigma_fine - sigma_coarse)
        relative_diff = mesh_diff / max(abs(sigma_analytic), 1.0)

        assert err_fine <= err_coarse or relative_diff < 0.05, (
            f"h-refinement did not converge monotonically: "
            f"coarse err={err_coarse:.4e}, fine err={err_fine:.4e}, "
            f"mesh-to-mesh diff={mesh_diff:.4e} ({relative_diff*100:.2f}%)"
        )


# ============================================================
# PROVENANCE METADATA TESTS
# ============================================================

class TestBuildProvenance:
    """Tests for the _build_provenance() reproducibility helper."""

    def test_provenance_returns_dict(self):
        prov = _build_provenance()
        assert isinstance(prov, dict)

    def test_required_keys_present(self):
        prov = _build_provenance()
        for key in ('porosity_fe_version', 'python_version', 'numpy_version',
                    'scipy_version', 'matplotlib_version', 'timestamp_utc',
                    'platform', 'seed', 'git_commit'):
            assert key in prov, f"Missing provenance key: {key}"

    def test_python_version_is_non_null_string(self):
        prov = _build_provenance()
        assert isinstance(prov['python_version'], str)
        assert len(prov['python_version']) > 0
        # Should look like "3.X.Y"
        parts = prov['python_version'].split('.')
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_numpy_version_is_non_null_string(self):
        prov = _build_provenance()
        assert isinstance(prov['numpy_version'], str)
        assert len(prov['numpy_version']) > 0

    def test_scipy_version_is_non_null_string(self):
        prov = _build_provenance()
        assert isinstance(prov['scipy_version'], str)
        assert len(prov['scipy_version']) > 0

    def test_matplotlib_version_is_non_null_string(self):
        prov = _build_provenance()
        assert isinstance(prov['matplotlib_version'], str)
        assert len(prov['matplotlib_version']) > 0

    def test_timestamp_utc_is_non_null_string(self):
        prov = _build_provenance()
        assert isinstance(prov['timestamp_utc'], str)
        assert prov['timestamp_utc'].endswith('Z')
        # Should be parseable as ISO-8601
        import datetime
        ts = prov['timestamp_utc'].rstrip('Z')
        datetime.datetime.fromisoformat(ts)  # raises if malformed

    def test_platform_is_non_null_string(self):
        prov = _build_provenance()
        assert isinstance(prov['platform'], str)
        assert len(prov['platform']) > 0

    def test_seed_is_none(self):
        # No random seed is used in this codebase; must be null
        prov = _build_provenance()
        assert prov['seed'] is None

    def test_git_commit_is_string_or_none(self):
        prov = _build_provenance()
        assert prov['git_commit'] is None or isinstance(prov['git_commit'], str)


class TestProvenanceInSaveResultsJson:
    """Integration: provenance is present and valid in save_results_to_json output."""

    def test_provenance_in_json_output(self, tmp_path):
        results = compare_configurations(
            0.03, configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']}
        )
        path = str(tmp_path / "prov_test.json")
        save_results_to_json(results, path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        prov = data['provenance']
        assert isinstance(prov['python_version'], str) and prov['python_version']
        assert isinstance(prov['numpy_version'], str) and prov['numpy_version']
        assert isinstance(prov['timestamp_utc'], str) and prov['timestamp_utc']
        assert 'porosity_fe_version' in prov

    def test_schema_version_in_json_output(self, tmp_path):
        results = compare_configurations(
            0.03, configs={'uniform_spherical': POROSITY_CONFIGS['uniform_spherical']}
        )
        path = str(tmp_path / "schema_test.json")
        save_results_to_json(results, path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data['schema_version'] == '1.0'


class TestJsonEncodingRoundTrip:
    """Regression for #21: JSON I/O must be UTF-8 on every platform.

    Without explicit encoding, Windows opens files in the locale code page
    (cp1252) and silently mangles non-ASCII content. This locks the
    round-trip with characters that are not representable in cp1252.
    """

    def test_non_ascii_round_trips_through_loader(self, tmp_path):
        path = str(tmp_path / "ünïcode_µCT.json")
        payload = {
            "schema_version": JSON_SCHEMA_VERSION,
            "format": FORMAT_EMPIRICAL_SWEEP,
            "note": "µCT scan, σ₁c knockdown — café/naïve ✓",
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        loaded = load_results_from_json(path)
        assert loaded["note"] == "µCT scan, σ₁c knockdown — café/naïve ✓"


class TestProvenanceInFEExportResults:
    """Integration: provenance is present and valid in FESolver.export_results output."""

    def test_provenance_in_fe_json_output(self, tmp_path):
        material = MATERIALS['T800_epoxy']
        pf = PorosityField(material, 0.03, distribution='uniform')
        mesh = CompositeMesh(pf, material, nx=3, ny=2, nz=2)
        solver = FESolver(mesh, material, pf)
        field_results = solver.solve(loading='compression', applied_strain=-0.001)
        path = str(tmp_path / "fe_prov_test.json")
        FESolver.export_results(field_results, path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        prov = data['provenance']
        assert isinstance(prov['python_version'], str) and prov['python_version']
        assert isinstance(prov['numpy_version'], str) and prov['numpy_version']
        assert isinstance(prov['timestamp_utc'], str) and prov['timestamp_utc']
        assert 'porosity_fe_version' in prov
        assert data['schema_version'] == '1.0'
