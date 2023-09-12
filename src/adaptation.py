# -*- coding: utf-8 -*-

from typing import Callable

import torch

from .gaussian_quadrature_1D import get_gaussian_points_physical_space
from .random_points import get_id_points

Function = Callable[[torch.Tensor], torch.Tensor]


def get_new_adapted_points(
    f: Function, input_points: torch.Tensor, n_points_out: int
) -> torch.Tensor:
    x = input_points
    t1 = torch.cat(
        (
            torch.cat(
                tuple(
                    map(
                        lambda gp: get_id_points(gp[0], gp[1], 7).reshape(-1),
                        zip(x[:-1], x[1:]),
                    )
                )
            ),
            x[1:-1],
        )
    ).unique()
    t1 = t1[(t1 != x[0]) & (t1 != x[-1])]
    # t1 = t1[t1 != 1.0]
    _, idx = f(t1.reshape(-1, 1)).reshape(-1).sort()
    t2 = t1[idx][:n_points_out]
    return t2.sort()[0]
