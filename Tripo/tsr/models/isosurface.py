from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import mcubes


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        raise NotImplementedError


class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self.mc_func = mcubes.marching_cubes
        self._grid_vertices: Optional[torch.FloatTensor] = None

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        if self._grid_vertices is None:
            # keep the vertices on CPU so that we can support very large resolution
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
            self._grid_vertices = verts
        return self._grid_vertices

    def forward(
        self,
        level: torch.FloatTensor,
    ):
        level = -level.view(
            self.resolution,
            self.resolution,
            self.resolution
        )

        volume = level.detach().cpu().numpy()

        vertices, triangles = self.mc_func(
            volume,
            0.0
        )

        vertices = vertices[:, [2, 1, 0]]
        vertices = vertices / (self.resolution - 1.0)

        vertices = torch.from_numpy(vertices).float().to(level.device)
        triangles = torch.from_numpy(triangles.astype(np.int64)).to(level.device)

        return vertices, triangles

