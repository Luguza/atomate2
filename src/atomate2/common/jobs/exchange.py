"""Jobs for fitting Heisenberg exchange parameters and building exchange docs.

These replace the atomate1 ``HeisenbergModelMapping``, ``HeisenbergModelToDb`` and
``VampireToDb`` firetasks. Outputs flow between jobs through jobflow references
rather than a MongoDB collection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

from jobflow import Flow, Maker, Response, job
from pymatgen.analysis.magnetism.heisenberg import HeisenbergMapper

from atomate2.common.jobs.magnetism import (
    enumerate_magnetic_orderings,
    run_ordering_calculations,
)
from atomate2.common.schemas.exchange import (
    ExchangeDocument,
    ExchangeFit,
    ExchangeRound,
)

if TYPE_CHECKING:
    from emmet.core.tasks import TaskDoc
    from pymatgen.analysis.magnetism.heisenberg import HeisenbergModel
    from pymatgen.core.structure import Structure

    from atomate2.vampire.schemas.vampire_output import VampireOutput

    # One round's static calculations, as run_ordering_calculations outputs them.
    RoundCalculations: TypeAlias = list[tuple[TaskDoc, dict, str]]

FIRST_CELL_SIZE = 1
# Ferromagnetic only in the first round: it is the same ordering in every cell size.
FIRST_ROUND_STRATEGIES = ("ferromagnetic", "antiferromagnetic")
LATER_ROUND_STRATEGIES = ("antiferromagnetic",)

# DECISION 1: stop on the leave-one-out error, the J change between rounds, or both.
StopRule = Literal["leave_one_out_error", "j_change", "both"]


@job(name="heisenberg mapping")
def heisenberg_mapping(
    structures: list[Structure],
    energies: list[float],
    parent: Structure | None = None,
    heisenberg_settings: dict | None = None,
) -> HeisenbergModel:
    """Fit a classical Heisenberg Hamiltonian to magnetic structures and energies.

    This wraps pymatgen's ``HeisenbergMapper``. Each exchange parameter ``J_ij`` is
    one symmetry orbit of bonds in the parent cell, fit by least squares over all
    supplied orderings.

    Parameters
    ----------
    structures : list[Structure]
        Magnetic structures, each carrying a "magmom" site property, in any order.
    energies : list[float]
        Energies **per atom** (eV) corresponding to each structure. These are
        converted to total energies internally, as required by HeisenbergMapper.
    parent : Structure or None
        The paramagnetic parent cell. If None, HeisenbergMapper infers it from the
        lowest-energy ordering and warns.
    heisenberg_settings : dict or None
        Keyword arguments for HeisenbergMapper, e.g. ``{"cutoff": 3.0}``.

    Returns
    -------
    HeisenbergModel
        The fitted, MSONable Heisenberg model.
    """
    heisenberg_settings = heisenberg_settings or {}
    total_energies = [e * len(s) for s, e in zip(structures, energies, strict=True)]
    hmapper = HeisenbergMapper(
        structures, total_energies, parent, **heisenberg_settings
    )
    return hmapper.get_heisenberg_model()


# --- Convergence loop over supercell rounds ---


@job(name="exchange round")
def run_exchange_round(
    structure: Structure,
    static_maker: Maker,
    default_magmoms: dict[str, float] | None,
    heisenberg_settings: dict,
    stop_rule: StopRule,
    j_tolerance: float,
    leave_one_out_tolerance: float,
    max_cell_size: int,
    cell_size: int,
    calculations: list[RoundCalculations],
    rounds: list[ExchangeRound],
) -> Response | ExchangeFit:
    """Fit the orderings so far on `structure`; return the fit or queue a new round."""
    # 1. If `calculations` is empty, return Response(replace=queue_next_round(...,
    #    cell_size=FIRST_CELL_SIZE)).
    # 2. structures, energies_per_atom = collect_orderings(calculations)
    # 3. check_orderings(structures, energies_per_atom)
    # 4. mapper = map_heisenberg(structures, energies_per_atom, structure,
    #    heisenberg_settings) (open: while too few orderings constrain any J,
    #    pymatgen raises ValueError; catch it and go to step 8?)
    # 5. model = mapper.get_heisenberg_model(); current = ExchangeRound(cell_size,
    #    len(structures), model.ex_params, model.residual,
    #    model.leave_one_out_error).
    # 6. If has_resolved_every_target(mapper) and has_converged(current, last of
    #    `rounds` or None, stop_rule, j_tolerance, leave_one_out_tolerance):
    #    return ExchangeFit(model, rounds + [current]).
    # 7. If cell_size >= max_cell_size: warn that the J's did not converge and
    #    return ExchangeFit(model, rounds + [current]).
    # 8. Return Response(replace=queue_next_round(..., cell_size + 1,
    #    calculations, rounds + [current])).
    raise NotImplementedError


def queue_next_round(
    structure: Structure,
    static_maker: Maker,
    default_magmoms: dict[str, float] | None,
    heisenberg_settings: dict,
    stop_rule: StopRule,
    j_tolerance: float,
    leave_one_out_tolerance: float,
    max_cell_size: int,
    cell_size: int,
    calculations: list[RoundCalculations],
    rounds: list[ExchangeRound],
) -> Flow:
    """Return a flow that runs the statics of the `cell_size` orderings, then fits."""
    # 1. enumerate_job = enumerate_magnetic_orderings(structure, default_magmoms,
    #    strategies=FIRST_ROUND_STRATEGIES if cell_size == FIRST_CELL_SIZE else
    #    LATER_ROUND_STRATEGIES, automatic=False, truncate_by_symmetry=False,
    #    transformation_kwargs={"min_cell_size": cell_size,
    #    "max_cell_size": cell_size}) (open: automatic=False drops pymatgen's extra
    #    strategies; right for fitting?)
    # 2. new_orderings_job = drop_known_orderings(enumerate_job.output, calculations)
    # 3. statics_job = run_ordering_calculations(new_orderings_job.output,
    #    static_maker, relax_maker=None)
    #    DECISION 4: commensurate k-point meshes per supercell plug in here, on the
    #    static inputs of each ordering.
    # 4. next_round_job = run_exchange_round(structure, static_maker, ...,
    #    cell_size, calculations + [statics_job.output], rounds)
    # 5. Return Flow([enumerate_job, new_orderings_job, statics_job,
    #    next_round_job], output=next_round_job.output).
    raise NotImplementedError


@job(name="drop known orderings")
def drop_known_orderings(
    orderings: tuple[list[Structure], list[str]],
    calculations: list[RoundCalculations],
) -> tuple[list[Structure], list[str]]:
    """Return `orderings` without those already computed in `calculations`."""
    # 1. known = the input structures, with magmoms, of every task in `calculations`.
    # 2. Keep each ordering whose CollinearMagneticStructureAnalyzer.matches_ordering
    #    is False for every known structure (open: does it match an ordering to
    #    the same ordering in a larger supercell?).
    # 3. Return the kept structures with their origins.
    raise NotImplementedError


# --- Fitting the orderings ---


def collect_orderings(
    calculations: list[RoundCalculations],
) -> tuple[list[Structure], list[float]]:
    """Return the magmom structures and energies per atom of all round calculations."""
    # 1. For each task of each round: ordering = MagneticOrderingRelaxation
    #    .from_task_document(task) (open: fit on the output moments, as atomate1
    #    did, or on the enumerated input moments?).
    # 2. Copy ordering.structure and add ordering.magmoms as the "magmom" site
    #    property.
    # 3. Return the structures and the ordering.energy_per_atom values.
    raise NotImplementedError


def check_orderings(
    structures: list[Structure], energies_per_atom: list[float]
) -> None:
    """Raise ValueError if the counts differ or a structure has no "magmom" property."""
    # 1. Raise if len(structures) != len(energies_per_atom).
    # 2. Raise if any structure lacks a "magmom" site property.
    raise NotImplementedError


def map_heisenberg(
    structures: list[Structure],
    energies_per_atom: list[float],
    parent: Structure | None,
    heisenberg_settings: dict,
) -> HeisenbergMapper:
    """Return a HeisenbergMapper fitted to the orderings and their energies per atom."""
    # 1. Convert the energies per atom to total energies.
    # 2. Return HeisenbergMapper(structures, total_energies, parent,
    #    **heisenberg_settings).
    raise NotImplementedError


# --- Convergence checks ---


def has_resolved_every_target(mapper: HeisenbergMapper) -> bool:
    """Return True if the fit constrains every J that the cutoff targets."""
    # 1. targets = the interaction labels in mapper.dists.
    # 2. fitted = the J columns of mapper.ex_mat (all but "E" and "E0").
    # 3. Return True if every target is fitted and the J and E0 columns of
    #    mapper.ex_mat have full column rank.
    raise NotImplementedError


def has_converged(
    current: ExchangeRound,
    previous: ExchangeRound | None,
    stop_rule: StopRule,
    j_tolerance: float,
    leave_one_out_tolerance: float,
) -> bool:
    """Return True if `current` meets `stop_rule` against `previous`."""
    # 1. Return False if previous is None.
    # 2. j_converged = largest_j_change(current.ex_params, previous.ex_params)
    #    <= j_tolerance (meV/muB^2).
    # 3. leave_one_out_converged = current.leave_one_out_error
    #    <= leave_one_out_tolerance (meV per magnetic ion). Absolute, not the change
    #    between rounds: the rounds' errors come from different sets of orderings.
    # 4. Return j_converged, leave_one_out_converged or both, as `stop_rule` says.
    raise NotImplementedError


def largest_j_change(
    current_ex_params: dict[str, float], previous_ex_params: dict[str, float]
) -> float:
    """Return the largest absolute J change (meV/muB^2) between two fits."""
    # 1. Take the J labels of both fits, without "E0".
    # 2. Return infinity if the label sets differ.
    # 3. Return the largest absolute difference over the labels.
    raise NotImplementedError


@job(name="build exchange doc")
def build_exchange_doc(
    heisenberg_model: HeisenbergModel,
    parent_structure: Structure | None = None,
    vampire_output: VampireOutput | None = None,
    vampire_settings: dict | None = None,
) -> ExchangeDocument:
    """Assemble the final ExchangeDocument from a fitted model and optional Tc run.

    Parameters
    ----------
    heisenberg_model : HeisenbergModel
        The fitted Heisenberg model from :func:`heisenberg_mapping`.
    parent_structure : Structure or None
        The paramagnetic parent cell the model was fitted on.
    vampire_output : VampireOutput or None
        The Vampire Monte-Carlo result, if the critical-temperature step was run.
    vampire_settings : dict or None
        The keyword arguments :func:`run_vampire` was called with, if it was run.

    Returns
    -------
    ExchangeDocument
        The final summary document.
    """
    return ExchangeDocument.from_model(
        heisenberg_model,
        parent_structure=parent_structure,
        vampire_output=vampire_output,
        vampire_settings=vampire_settings,
    )
