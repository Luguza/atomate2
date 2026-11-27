"""Job for running Vampire Monte-Carlo on a fitted Heisenberg model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Maker, job

from atomate2.vampire.vampire_caller import VampireCaller

if TYPE_CHECKING:
    from jobflow import Job
    from pymatgen.analysis.magnetism.heisenberg import HeisenbergModel

    from atomate2.vampire.schemas.vampire_output import VampireOutput


@job(name="run vampire")
def run_vampire(
    heisenberg_model: HeisenbergModel,
    mc_settings: dict | None = None,
) -> VampireOutput:
    """Run Vampire Monte-Carlo to estimate the critical temperature.

    This wraps the (vendored) ``VampireCaller``, which shells out to the external
    VAMPIRE binary named by ``ATOMATE2_VAMPIRE_CMD`` (``vampire-serial`` by
    default; set e.g. ``srun -n 24 vampire-parallel`` to run the MPI build). A
    clear error is raised if the binary is not found on PATH.

    Parameters
    ----------
    heisenberg_model : HeisenbergModel
        The fitted Heisenberg model.
    mc_settings : dict or None
        Keyword arguments for VampireCaller, e.g. ``mc_box_size``,
        ``equil_timesteps``, ``mc_timesteps``.

    Returns
    -------
    VampireOutput
        The Vampire Monte-Carlo result, exposing ``critical_temp``.
    """
    mc_settings = mc_settings or {}
    vampire_caller = VampireCaller(hm=heisenberg_model, **mc_settings)
    return vampire_caller.output


@dataclass
class VampireMaker(Maker):
    """Maker for a Vampire Monte-Carlo run on a fitted Heisenberg model.

    Parameters
    ----------
    name : str
        Name of the jobs produced by this Maker.
    mc_settings : dict
        Keyword arguments for VampireCaller, e.g. ``mc_box_size``,
        ``equil_timesteps``, ``mc_timesteps``.
    """

    name: str = "run vampire"
    mc_settings: dict = field(default_factory=dict)

    def make(self, heisenberg_model: HeisenbergModel) -> Job:
        """Return a job that runs Vampire on `heisenberg_model`."""
        # 1. Return run_vampire(heisenberg_model, self.mc_settings), renamed to
        #    self.name (open: once pymatgen #4786 lands, pymatgen writes and parses
        #    the files and this job only runs SETTINGS.VAMPIRE_CMD).
        raise NotImplementedError
