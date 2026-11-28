"""Flow that fits magnetic exchange parameters from its own static calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

from atomate2.common.jobs.exchange import (
    ATOM_COST_WEIGHT,
    MAX_CELL_SIZE,
    ORDERINGS_PER_PARAMETER,
    build_exchange_doc,
    check_magnetic_sites,
    enumerate_candidates,
    run_exchange_round,
    symmetrize_to_paramagnetic_parent,
)
from atomate2.vasp.jobs.core import StaticMaker
from atomate2.vasp.sets.core import StaticSetGenerator

if TYPE_CHECKING:
    from jobflow import Job
    from pymatgen.core.structure import Structure

    from atomate2.common.flows.magnetism import MagneticOrderingsMaker
    from atomate2.vampire.jobs.run_vampire import VampireMaker


__all__ = ["ExchangeMaker"]


@dataclass
class ExchangeMaker(Maker):
    """Maker to fit Heisenberg exchange parameters on one fixed geometry.

    The flow enumerates the magnetic orderings of the paramagnetic parent cell in
    supercells of up to ``max_magnetic_sites``, and fits a classical Heisenberg
    Hamiltonian to their static energies with pymatgen's ``HeisenbergMapper``. Each
    round runs statics only for the orderings that add the most information to the
    fit per atom (D-optimal design), until every J within the cutoff is resolved
    and the fit's leave-one-out error is at most ``leave_one_out_tolerance``, or
    until the candidates run out or none is worth its atoms. Each static is fitted
    with its output moments. If the candidates cannot resolve every J within the
    cutoff, the flow raises before any static runs.

    Besides ``leave_one_out_tolerance`` and ``max_magnetic_sites``, three fixed
    limits shape the loop: supercells hold at most ``MAX_CELL_SIZE`` parent cells, a
    round selects at most ``ORDERINGS_PER_PARAMETER`` orderings per fitted
    parameter, and a candidate must gain more than ``ATOM_COST_WEIGHT`` per parent
    cell's worth of atoms. The output document records all five in
    ``loop_limits``, and in ``stop_reason`` why the loop ended.

    Convergence refers to the J's within the cutoff, which assumes the exchange
    beyond it is negligible. To check that, run again with a larger ``cutoff`` in
    ``heisenberg_settings`` and compare the J's.

    Originally implemented in atomate (v1) for VASP as the ``ExchangeWF``.

    Parameters
    ----------
    name : str
        Name of the flows produced by this Maker.
    static_maker : Maker
        Maker for the static calculation of each ordering. VASP is the default.
    ground_state_maker : MagneticOrderingsMaker | None
        If given, runs first to find the magnetic ground state with relaxations; the
        loop then fits on that ground state. If None, it fits on the input
        structure. Either is symmetrized to its paramagnetic space group first, at
        ``PARENT_SYMPREC``, which removes small distortions of the ordered phase.
    heisenberg_settings : dict
        Keyword arguments for pymatgen's HeisenbergMapper, e.g. ``cutoff`` in
        Angstrom, which fixes the J's the fit targets.
    default_magmoms : dict[str, float] | None
        Initial moments (muB) of the elements that can carry a moment. If None,
        pymatgen's ``default_magmoms.yaml`` is used.
    leave_one_out_tolerance : float
        Largest accepted leave-one-out error of the fit, in meV per magnetic ion.
    max_magnetic_sites : int
        Most magnetic sites a supercell may hold for its orderings to be enumerated.
        Every supercell shape up to that size is included. The number of orderings
        grows about exponentially with it.
    vampire_maker : VampireMaker | None
        If given, runs Vampire Monte-Carlo on the fitted model for the critical
        temperature.
    """

    name: str = "exchange"
    # Tight SCF: the J's come from energy differences of a few meV per magnetic ion.
    static_maker: Maker = field(
        default_factory=lambda: StaticMaker(
            input_set_generator=StaticSetGenerator(user_incar_settings={"EDIFF": 1e-7})
        )
    )
    ground_state_maker: MagneticOrderingsMaker | None = None
    heisenberg_settings: dict = field(default_factory=dict)
    default_magmoms: dict[str, float] | None = None
    leave_one_out_tolerance: float = 1.0
    max_magnetic_sites: int = 16
    vampire_maker: VampireMaker | None = None

    def make(self, structure: Structure) -> Flow:
        """Return a flow that fits exchange parameters on `structure`."""
        # The parent below is only a job reference, so enumerate_candidates repeats
        # the site check on it.
        self._check_inputs(structure)
        jobs: list[Job | Flow] = []

        geometry = structure
        if self.ground_state_maker is not None:
            search_flow = self.ground_state_maker.make(structure)
            jobs.append(search_flow)
            geometry = search_flow.output.ground_state_structure
        parent_job = symmetrize_to_paramagnetic_parent(geometry)
        jobs.append(parent_job)
        parent = parent_job.output

        heisenberg_settings = {
            **self.heisenberg_settings,
            "default_magmoms": self.default_magmoms,
        }
        candidates_job = enumerate_candidates(
            parent, heisenberg_settings, self.max_magnetic_sites
        )
        jobs.append(candidates_job)

        # Replaces itself round by round, so its output is the final ExchangeFit.
        round_job = run_exchange_round(
            parent,
            self.static_maker,
            heisenberg_settings,
            self.leave_one_out_tolerance,
            candidates=candidates_job.output,
            calculations=[],
            rounds=[],
        )
        jobs.append(round_job)
        heisenberg_model = round_job.output.heisenberg_model

        vampire_output = vampire_settings = None
        if self.vampire_maker is not None:
            vampire_job = self.vampire_maker.make(heisenberg_model)
            jobs.append(vampire_job)
            vampire_output = vampire_job.output
            vampire_settings = self.vampire_maker.mc_settings

        doc_job = build_exchange_doc(
            heisenberg_model,
            parent_structure=parent,
            rounds=round_job.output.rounds,
            stop_reason=round_job.output.stop_reason,
            loop_limits={
                "leave_one_out_tolerance": self.leave_one_out_tolerance,
                "max_magnetic_sites": self.max_magnetic_sites,
                "max_cell_size": MAX_CELL_SIZE,
                "orderings_per_parameter": ORDERINGS_PER_PARAMETER,
                "atom_cost_weight": ATOM_COST_WEIGHT,
            },
            vampire_output=vampire_output,
            vampire_settings=vampire_settings,
        )
        jobs.append(doc_job)

        return Flow(
            jobs,
            output=doc_job.output,
            name=f"{self.name} ({structure.composition.reduced_formula})",
        )

    def _check_inputs(self, structure: Structure) -> None:
        """Raise ValueError if the settings or `structure` cannot start the loop."""
        if self.leave_one_out_tolerance <= 0:
            raise ValueError(
                "leave_one_out_tolerance must be positive, got "
                f"{self.leave_one_out_tolerance}."
            )
        if "default_magmoms" in self.heisenberg_settings:
            raise ValueError(
                "Set default_magmoms on the maker, not in heisenberg_settings: make "
                "would overwrite it there."
            )
        check_magnetic_sites(structure, self.default_magmoms, self.max_magnetic_sites)
