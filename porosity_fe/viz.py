"""Publication-quality plotting helpers (matplotlib)."""

import logging

import matplotlib.pyplot as plt
import numpy as np

from ._style import (
    LABEL_KNOCKDOWN,
    LABEL_POROSITY_PCT,
    LABEL_SCF,
    LABEL_STIFFNESS_RETENTION_FRAC,
    LABEL_X_MM,
    LABEL_Y_MM,
    LABEL_Z_MM,
)
from .mesh import CompositeMesh
from .porosity_field import PorosityField
from .void_geometry import VoidGeometry

logger = logging.getLogger("porosity_fe_analysis")

# ============================================================
# SECTION 7: VISUALIZATION
# ============================================================

class FEVisualizer:
    """Publication-quality plotting for porosity analysis."""

    @staticmethod
    def plot_porosity_field(porosity_field: PorosityField, save_path: str = None):
        """Single panel: through-thickness porosity profile."""
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        z, Vp = porosity_field.effective_porosity_profile(nz=200)
        ax.plot(Vp * 100, z, 'b-', linewidth=2)
        ax.set_xlabel(LABEL_POROSITY_PCT)
        ax.set_ylabel(LABEL_Z_MM)
        ax.set_title('Through-Thickness Porosity Profile')
        ax.set_xlim(left=0)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logger.info("Saved: %s", save_path)
        return fig

    @staticmethod
    def plot_mesh_3d(mesh: CompositeMesh, save_path: str = None):
        """3D hex mesh wireframe with void elements highlighted."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot top and bottom surface grids
        nx, ny = mesh.nx, mesh.ny
        n_per_layer = (nx + 1) * (ny + 1)

        for layer_idx in [0, mesh.nz]:
            start = layer_idx * n_per_layer
            end = start + n_per_layer
            layer_nodes = mesh.nodes[start:end]
            X = layer_nodes[:, 0].reshape(ny + 1, nx + 1)
            Y = layer_nodes[:, 1].reshape(ny + 1, nx + 1)
            Z = layer_nodes[:, 2].reshape(ny + 1, nx + 1)
            ax.plot_wireframe(X, Y, Z, alpha=0.3, color='gray', linewidth=0.5)

        # Highlight void elements as red wireframe hex boxes
        hex_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
        ]
        if len(mesh.void_elements) > 0:
            for eidx in mesh.void_elements[:50]:  # limit for performance
                corners = mesh.nodes[mesh.elements[eidx]]  # (8, 3)
                for i1, i2 in hex_edges:
                    ax.plot3D(
                        *zip(corners[i1], corners[i2]),
                        color='red', linewidth=1.5, alpha=0.8, zorder=6,
                    )

        ax.set_xlabel(LABEL_X_MM)
        ax.set_ylabel(LABEL_Y_MM)
        ax.set_zlabel(LABEL_Z_MM)
        ax.set_title('3D Mesh with Porosity')

        if save_path:
            plt.savefig(save_path)
            logger.info("Saved: %s", save_path)
        return fig

    @staticmethod
    def plot_mesh_detail(mesh: CompositeMesh, save_path: str = None):
        """Cross-section with porosity contour + single hex element."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: cross-section at mid-y
        ny_mid = mesh.ny // 2
        nx1 = mesh.nx + 1
        ny1 = mesh.ny + 1
        indices = []
        for k in range(mesh.nz + 1):
            for i in range(mesh.nx + 1):
                idx = k * ny1 * nx1 + ny_mid * nx1 + i
                indices.append(idx)
        indices = np.array(indices)  # type: ignore[assignment]  # list rebound to ndarray
        X = mesh.nodes[indices, 0].reshape(mesh.nz + 1, mesh.nx + 1)
        Z = mesh.nodes[indices, 2].reshape(mesh.nz + 1, mesh.nx + 1)
        P = mesh.porosity[indices].reshape(mesh.nz + 1, mesh.nx + 1)

        im = axes[0].contourf(X, Z, P * 100, levels=20, cmap='YlOrRd')
        plt.colorbar(im, ax=axes[0], label=LABEL_POROSITY_PCT)
        axes[0].set_xlabel(LABEL_X_MM)
        axes[0].set_ylabel(LABEL_Z_MM)
        axes[0].set_title('Cross-Section Porosity')
        axes[0].set_aspect('equal')

        # Right: single hex element diagram
        ax = axes[1]
        corners = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=float)
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
        for e in edges:
            pts = corners[list(e)]
            ax.plot(pts[:, 0] + pts[:, 1]*0.3, pts[:, 2] + pts[:, 1]*0.3,
                   'b-', linewidth=1.5)
        for idx, c in enumerate(corners):
            ax.plot(c[0] + c[1]*0.3, c[2] + c[1]*0.3, 'ko', markersize=6)
            ax.annotate(str(idx), (c[0] + c[1]*0.3 + 0.05, c[2] + c[1]*0.3 + 0.05),
                       fontweight='bold')
        ax.set_title('8-Node Hexahedral Element')
        # Use the same (mm) units as the sibling cross-section panel so the
        # hex-element diagram is not ambiguous within the same figure (#53).
        ax.set_xlabel(LABEL_X_MM)
        ax.set_ylabel(LABEL_Z_MM)
        ax.set_aspect('equal')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logger.info("Saved: %s", save_path)
        return fig

    @staticmethod
    def plot_damage_contour(mesh: CompositeMesh, solver, save_path: str = None):
        """2D stiffness reduction map at midplane."""
        fig, ax = plt.subplots(figsize=(10, 4))

        # Get midplane slice
        nz_mid = mesh.nz // 2
        nx1 = mesh.nx + 1
        ny1 = mesh.ny + 1
        start = nz_mid * ny1 * nx1
        end = start + ny1 * nx1
        X = mesh.nodes[start:end, 0].reshape(ny1, nx1)
        Y = mesh.nodes[start:end, 1].reshape(ny1, nx1)

        if solver.nodal_knockdown is not None:
            kd = solver.nodal_knockdown[start:end].reshape(ny1, nx1)
        else:
            kd = mesh.stiffness_reduction[start:end].reshape(ny1, nx1)

        im = ax.contourf(X, Y, kd, levels=20, cmap='cividis')
        # GUI version uses "Stiffness Retention (%)"; static PNG was using
        # "Stiffness Retention (fraction)" and a 0..1 scale. The two paths
        # plot the same physical quantity (``stiffness_reduction`` is a
        # 0..1 retention fraction), so report it consistently as a
        # percentage and the GUI/PNG units cannot drift again (#53).
        plt.colorbar(im, ax=ax, label=LABEL_STIFFNESS_RETENTION_FRAC)
        ax.set_xlabel(LABEL_X_MM)
        ax.set_ylabel(LABEL_Y_MM)
        ax.set_title('Stiffness Reduction at Midplane')
        ax.set_aspect('equal')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logger.info("Saved: %s", save_path)
        return fig

    @staticmethod
    def plot_void_scf(void_geometry: VoidGeometry, save_path: str = None):
        """Stress concentration field around a single void."""
        fig, ax = plt.subplots(figsize=(8, 8))

        r_max = 3 * max(void_geometry.radii)
        x = np.linspace(-r_max, r_max, 200)
        y = np.linspace(-r_max, r_max, 200)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        dist = void_geometry.distance_field(X.ravel(), Y.ravel(), Z.ravel())
        dist = dist.reshape(X.shape)

        scf = void_geometry.stress_concentration_factor()
        scf_max = scf['compression']
        field = np.where(dist < 0, 0, 1.0 + (scf_max - 1) * np.exp(-dist / max(void_geometry.radii)))

        im = ax.contourf(X, Y, field, levels=30, cmap='magma')
        plt.colorbar(im, ax=ax, label=LABEL_SCF)
        ax.set_xlabel(LABEL_X_MM)
        ax.set_ylabel(LABEL_Y_MM)
        ax.set_title(
            f'SCF Field (aspect ratio={void_geometry.aspect_ratio:.1f})'
        )
        ax.set_aspect('equal')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logger.info("Saved: %s", save_path)
        return fig

    @staticmethod
    def plot_knockdown_curves(results_by_porosity: dict, save_path: str = None):
        """Strength vs porosity % for all loading modes."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        modes = ['compression', 'tension', 'shear', 'ilss']
        colors = {'judd_wright': 'blue', 'power_law': 'red', 'linear': 'green'}

        for idx, mode in enumerate(modes):
            ax = axes[idx]
            Vp_vals = sorted([float(k.replace('pct', '')) for k in results_by_porosity.keys()])

            for config_name in list(list(results_by_porosity.values())[0].keys()):
                # Empirical models
                for model in ['judd_wright', 'power_law', 'linear']:
                    kd_vals = []
                    for Vp_label in sorted(results_by_porosity.keys()):
                        r = results_by_porosity[Vp_label][config_name]['empirical']
                        kd_vals.append(r[mode][model]['knockdown'])
                    ax.plot(Vp_vals, kd_vals, color=colors[model],
                           linestyle='-' if 'uniform' in config_name else '--',
                           alpha=0.7, linewidth=1.5)

            ax.set_xlabel(LABEL_POROSITY_PCT)
            ax.set_ylabel(LABEL_KNOCKDOWN)
            ax.set_title(mode.upper())
            ax.set_ylim(0, 1.1)

        plt.suptitle('Porosity Knockdown Curves')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logger.info("Saved: %s", save_path)
        return fig

    @staticmethod
    def plot_model_comparison(results: dict, save_path: str = None):
        """Empirical model comparison bar chart."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        configs = list(results.keys())
        x = np.arange(len(configs))
        width = 0.2

        # Left: compression knockdown
        for i, model in enumerate(['judd_wright', 'power_law', 'linear']):
            vals = [results[c]['empirical']['compression'][model]['knockdown'] for c in configs]
            axes[0].bar(x + i * width, vals, width, label=model.replace('_', ' ').title())
        axes[0].set_xticks(x + width)
        # Tick labels intentionally smaller than rcParams.xtick.labelsize
        # because the config names are long and wrap to two lines.
        axes[0].set_xticklabels(
            [c.replace('_', '\n') for c in configs], fontsize=8,
        )
        axes[0].set_ylabel(LABEL_KNOCKDOWN)
        axes[0].set_title('Compression')
        axes[0].legend()
        axes[0].grid(True, axis='y')

        # Right: ILSS knockdown
        for i, model in enumerate(['judd_wright', 'power_law', 'linear']):
            vals = [results[c]['empirical']['ilss'][model]['knockdown'] for c in configs]
            axes[1].bar(x + i * width, vals, width, label=model.replace('_', ' ').title())
        axes[1].set_xticks(x + width)
        axes[1].set_xticklabels(
            [c.replace('_', '\n') for c in configs], fontsize=8,
        )
        axes[1].set_ylabel(LABEL_KNOCKDOWN)
        axes[1].set_title('ILSS')
        axes[1].legend()
        axes[1].grid(True, axis='y')

        plt.suptitle('Model Comparison')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logger.info("Saved: %s", save_path)
        return fig
