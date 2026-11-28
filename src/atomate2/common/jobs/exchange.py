"""Jobs of the exchange flow: candidates, the selecting and fitting loop, the doc."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from jobflow import Flow, Maker, Response, job
from pymatgen.analysis.magnetism.analyzer import (
    CollinearMagneticStructureAnalyzer,
    MagneticStructureEnumerator,
)
from pymatgen.analysis.magnetism.heisenberg import (
    DEFAULT_SYMPREC,
    ComputedOrdering,
    HeisenbergMapper,
    ParentCell,
)
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate2.common.jobs.magnetism import run_ordering_calculations
from atomate2.common.schemas.exchange import (
    ExchangeDocument,
    ExchangeFit,
    ExchangeRound,
)

if TYPE_CHECKING:
    from emmet.core.tasks import TaskDoc
    from pymatgen.analysis.magnetism.heisenberg import HeisenbergModel
    from pymatgen.core.structure import Structure

    from atomate2.common.schemas.exchange import StopReason
    from atomate2.vampire.schemas.vampire_output import VampireOutput

    # One round's static calculations, as run_ordering_calculations outputs them.
    RoundCalculations: TypeAlias = list[tuple[TaskDoc, dict, str]]
    # Ordered structures with their origins (e.g. "fm", "afm").
    Orderings: TypeAlias = tuple[list[Structure], list[str]]

# Limits of the loop; every ExchangeDocument records them in loop_limits.
# Largest supercell, in parent cells, to enumerate; pymatgen keeps the orderings of
# the smaller cells anyway, so larger ones only cost enumeration time.
MAX_CELL_SIZE = 8
# A round selects up to this many orderings per fitted parameter (E0 and the J's).
ORDERINGS_PER_PARAMETER = 2
# Cost of a candidate, in units of log det(X^T X), per parent cell's worth of atoms
# in its supercell; it is selected only if its information gain exceeds that.
ATOM_COST_WEIGHT = 0.03

# Ridge on X^T X, so log det is finite before the orderings fix every J.
SELECTION_PRIOR = 1e-6
# Loose enough to undo the small distortion an ordered phase leaves in the geometry.
PARENT_SYMPREC = 0.1  # Angstrom


# --- Reference geometry ---


@job(name="symmetrize to paramagnetic parent")
def symmetrize_to_paramagnetic_parent(structure: Structure) -> Structure:
    """Return `structure` without moments, symmetrized to its paramagnetic cell.

    Experimental and relaxed geometries often keep a small distortion of the ordered
    phase, which would split the magnetic sublattices and change the targeted J's.
    """
    structure = structure.copy()
    # Moments, as site property or species spin, would lower the symmetry.
    if "magmom" in structure.site_properties:
        structure.remove_site_property("magmom")
    structure.remove_spin()
    analyzer = SpacegroupAnalyzer(structure, symprec=PARENT_SYMPREC)
    return analyzer.get_primitive_standard_structure()


# --- Candidate orderings ---


@job(name="enumerate candidate orderings")
def enumerate_candidates(
    structure: Structure,
    heisenberg_settings: dict,
    max_magnetic_sites: int,
) -> Orderings:
    """Return every ordering of `structure` in supercells of up to `max_magnetic_sites`.

    The cap is on magnetic sites, not volume: the number of orderings grows about
    exponentially with them. Raises ValueError if the orderings cannot resolve
    every J the settings target, so a loop that cannot converge costs no DFT.
    """
    default_magmoms = heisenberg_settings["default_magmoms"]
    magnetic_sites = check_magnetic_sites(
        structure, default_magmoms, max_magnetic_sites
    )

    # truncate_by_symmetry=False: low-symmetry orderings often separate J's that
    # symmetric ones cannot. One call over all cell sizes, so a strategy without
    # orderings in the smallest cell finds them in larger ones.
    enumerator = MagneticStructureEnumerator(
        structure,
        default_magmoms=default_magmoms,
        automatic=True,
        truncate_by_symmetry=False,
        transformation_kwargs={
            "max_cell_size": min(MAX_CELL_SIZE, max_magnetic_sites // magnetic_sites)
        },
    )

    rows = selection_rows(structure, enumerator.ordered_structures, heisenberg_settings)
    fixable_parameters = np.linalg.matrix_rank(rows)
    num_parameters = rows.shape[1]
    if fixable_parameters < num_parameters:
        raise ValueError(
            f"The {len(rows)} candidate orderings can fix only {fixable_parameters} "
            f"of the {num_parameters} parameters (E0 and the J's within the cutoff), "
            "so the loop cannot converge. Lower the cutoff in heisenberg_settings or "
            "raise max_magnetic_sites."
        )
    return enumerator.ordered_structures, enumerator.ordered_structure_origins


def check_magnetic_sites(
    structure: Structure,
    default_magmoms: dict[str, float] | None,
    max_magnetic_sites: int,
) -> int:
    """Return the magnetic sites of the primitive `structure`, if it can be enumerated.

    Raises ValueError if it has none, or more than `max_magnetic_sites`.
    """
    magnetic_sites = magnetic_analyzer(
        structure, default_magmoms
    ).number_of_magnetic_sites
    if magnetic_sites == 0:
        raise ValueError(
            f"{structure.reduced_formula} has no magnetic site under default_magmoms."
        )
    if magnetic_sites > max_magnetic_sites:
        raise ValueError(
            f"The primitive cell of {structure.reduced_formula} has {magnetic_sites} "
            f"magnetic sites, more than max_magnetic_sites={max_magnetic_sites}, so "
            "not even its own orderings are enumerated. Raise max_magnetic_sites."
        )
    return magnetic_sites


def magnetic_analyzer(
    structure: Structure, default_magmoms: dict[str, float] | None
) -> CollinearMagneticStructureAnalyzer:
    """Return an analyzer of primitive `structure`, moments set by `default_magmoms`."""
    # By keyword: the second positional argument is overwrite_magmom_mode.
    return CollinearMagneticStructureAnalyzer(
        structure, overwrite_magmom_mode="replace_all", default_magmoms=default_magmoms
    )


# --- Convergence loop over rounds of selected orderings ---


@job(name="exchange round")
def run_exchange_round(
    structure: Structure,
    static_maker: Maker,
    heisenberg_settings: dict,
    leave_one_out_tolerance: float,
    candidates: Orderings,
    calculations: list[RoundCalculations],
    rounds: list[ExchangeRound],
) -> Response | ExchangeFit:
    """Fit the orderings so far on `structure`; return the fit or queue a new round."""
    known_structures: list[Structure] = []
    model: HeisenbergModel | None = None
    if calculations:
        known_structures, energies = collect_orderings(calculations)
        # Not caught: pymatgen raises only for fewer than 2 distinct orderings or no
        # J at all, and the first round selects at least one ordering per parameter.
        mapper = HeisenbergMapper(
            known_structures, energies, structure, **heisenberg_settings
        )
        model = mapper.get_heisenberg_model()
        rounds = [*rounds, summarize_round(model, calculations[-1])]

        # pymatgen drops a J the orderings cannot separate with only a warning, and
        # the fit of the remaining J's can still meet the tolerance.
        every_target_resolved = set(mapper.dists) <= set(mapper.ex_mat.columns)
        if (
            every_target_resolved
            and model.leave_one_out_error <= leave_one_out_tolerance
        ):
            return ExchangeFit(
                heisenberg_model=model, rounds=rounds, stop_reason="converged"
            )

    selected, remaining = select_orderings(
        structure, candidates, known_structures, heisenberg_settings
    )
    if not selected[0]:
        if model is None:
            raise ValueError("No candidate ordering to start the loop with.")
        stop_reason: StopReason = (
            "not_worth_its_atoms" if remaining[0] else "candidates_exhausted"
        )
        warnings.warn(
            f"The J's did not converge ({stop_reason}): the leave-one-out error is "
            f"{model.leave_one_out_error:.3g} meV per magnetic ion, above "
            f"leave_one_out_tolerance={leave_one_out_tolerance}. If the DFT noise is "
            "near the tolerance, raise leave_one_out_tolerance; if exchange beyond "
            "the cutoff may matter, rerun with a larger cutoff in "
            "heisenberg_settings; if the candidates ran out, raise "
            "max_magnetic_sites.",
            stacklevel=2,
        )
        return ExchangeFit(
            heisenberg_model=model, rounds=rounds, stop_reason=stop_reason
        )

    # DECISION 4: commensurate k-point meshes per supercell plug in here, on the
    # static inputs of each ordering.
    statics_job = run_ordering_calculations(selected, static_maker, relax_maker=None)
    next_round_job = run_exchange_round(
        structure,
        static_maker,
        heisenberg_settings,
        leave_one_out_tolerance,
        candidates=remaining,
        calculations=[*calculations, statics_job.output],
        rounds=rounds,
    )
    return Response(
        replace=Flow([statics_job, next_round_job], output=next_round_job.output)
    )


def summarize_round(
    model: HeisenbergModel, last_round: RoundCalculations
) -> ExchangeRound:
    """Return the summary of `model`, fitted after `last_round` added its orderings."""
    return ExchangeRound(
        new_ordering_atoms=[
            len(task.structure) for task, _metadata, _uuid in last_round
        ],
        num_orderings=len(model.structures),
        ex_params=model.ex_params,
        residual=model.residual,
        leave_one_out_error=model.leave_one_out_error,
    )


# --- Selecting informative orderings ---


def select_orderings(
    structure: Structure,
    candidates: Orderings,
    known_structures: list[Structure],
    heisenberg_settings: dict,
) -> tuple[Orderings, Orderings]:
    """Pick the candidates that add the most information per atom to the fit.

    Greedy D-optimal design: each pick maximizes the gain in log det(X^T X) minus
    its atom cost. Returns the selected and the remaining candidates.
    """
    candidate_structures, _origins = candidates
    rows = selection_rows(
        structure, [*known_structures, *candidate_structures], heisenberg_settings
    )
    known_rows, candidate_rows = np.split(rows, [len(known_structures)])
    num_parameters = rows.shape[1]
    information = known_rows.T @ known_rows + SELECTION_PRIOR * np.identity(
        num_parameters
    )
    parent_cells = [
        len(candidate) / len(structure) for candidate in candidate_structures
    ]

    remaining = list(range(len(candidate_structures)))
    selected: list[int] = []
    while remaining and len(selected) < ORDERINGS_PER_PARAMETER * num_parameters:
        scores = [
            np.log1p(
                candidate_rows[i] @ np.linalg.solve(information, candidate_rows[i])
            )
            - ATOM_COST_WEIGHT * parent_cells[i]
            for i in remaining
        ]
        best = int(np.argmax(scores))
        if scores[best] <= 0:
            break
        chosen = remaining.pop(best)
        selected.append(chosen)
        information += np.outer(candidate_rows[chosen], candidate_rows[chosen])

    return take(candidates, selected), take(candidates, remaining)


def take(orderings: Orderings, indices: list[int]) -> Orderings:
    """Return the orderings at `indices`, with their origins."""
    structures, origins = orderings
    return [structures[i] for i in indices], [origins[i] for i in indices]


def selection_rows(
    structure: Structure, orderings: list[Structure], heisenberg_settings: dict
) -> np.ndarray:
    """Return one design-matrix row per ordering: 1 for E0, then its spin signs per J.

    Every J the settings target gets a column, shortest first, also those no ordering
    resolves yet. Signs, because the moments are unknown before the static.
    """
    default_magmoms = heisenberg_settings["default_magmoms"]
    magnetic_species = {
        species.symbol
        for species in magnetic_analyzer(
            structure, default_magmoms
        ).types_of_magnetic_species
    }
    parent = ParentCell(
        structure,
        magnetic_species,
        cutoff=heisenberg_settings.get("cutoff", 0),
        symprec=heisenberg_settings.get("symprec", DEFAULT_SYMPREC),
    )
    matcher = heisenberg_settings.get("matcher") or StructureMatcher(
        primitive_cell=False, attempt_supercell=True
    )

    rows = []
    for ordering in orderings:
        # The energy is unused: only the spin products are read.
        computed = ComputedOrdering(
            ordering, 0.0, parent, matcher, default_magmoms=default_magmoms
        )
        rows.append([1.0, *computed.spin_products(signs_only=True).values()])
    return np.array(rows).reshape(len(orderings), 1 + len(parent.interactions))


# --- Fitting the orderings ---


def collect_orderings(
    calculations: list[RoundCalculations],
) -> tuple[list[Structure], list[float]]:
    """Return the output structures, with their moments, and their total energies.

    Fitted on the output moments, a static that flipped spins or lost moment counts
    as the state it converged to.
    """
    tasks = [
        task for round_tasks in calculations for task, _metadata, _uuid in round_tasks
    ]
    return [task.structure for task in tasks], [task.output.energy for task in tasks]


@job(name="build exchange doc")
def build_exchange_doc(
    heisenberg_model: HeisenbergModel,
    parent_structure: Structure,
    rounds: list[ExchangeRound] | None = None,
    stop_reason: StopReason | None = None,
    loop_limits: dict[str, float] | None = None,
    vampire_output: VampireOutput | None = None,
    vampire_settings: dict | None = None,
) -> ExchangeDocument:
    """Assemble the final ExchangeDocument from a fitted model and optional Tc run.

    Parameters
    ----------
    heisenberg_model : HeisenbergModel
        The fitted Heisenberg model of the convergence loop.
    parent_structure : Structure
        The paramagnetic parent cell the model was fitted on.
    rounds : list[ExchangeRound] or None
        The fit summary of each round of the convergence loop.
    stop_reason : StopReason or None
        Why the convergence loop ended.
    loop_limits : dict[str, float] or None
        The limits of the convergence loop the fit ran with.
    vampire_output : VampireOutput or None
        The Vampire Monte-Carlo result, if the critical-temperature step was run.
    vampire_settings : dict or None
        The Monte-Carlo settings of the VampireMaker, if Vampire was run.

    Returns
    -------
    ExchangeDocument
        The final summary document.
    """
    return ExchangeDocument.from_model(
        heisenberg_model,
        parent_structure=parent_structure,
        rounds=rounds,
        stop_reason=stop_reason,
        loop_limits=loop_limits,
        vampire_output=vampire_output,
        vampire_settings=vampire_settings,
    )
