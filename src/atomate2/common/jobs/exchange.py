"""Jobs of the exchange flow: candidates, the selecting and fitting loop, the doc."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from jobflow import Flow, Maker, Response, job
from pymatgen.analysis.magnetism.heisenberg import HeisenbergMapper

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
    # 1. structure = a copy of `structure`.
    # 2. Remove its "magmom" site property, if any, so the spins no longer lower the
    #    symmetry.
    # 3. Return SpacegroupAnalyzer(structure, symprec=PARENT_SYMPREC)
    #    .get_primitive_standard_structure(): the paramagnetic space group, as the
    #    primitive cell.
    raise NotImplementedError


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
    # 1. default_magmoms = heisenberg_settings["default_magmoms"], the one source
    #    the fit and the selection use too.
    # 2. magnetic_sites = count_magnetic_sites(structure, default_magmoms); raise
    #    ValueError if it is 0 or above max_magnetic_sites (a ground-state parent
    #    is only known here).
    # 3. largest_cell_size = min(MAX_CELL_SIZE, max_magnetic_sites // magnetic_sites).
    # 4. enumerator = MagneticStructureEnumerator(structure, default_magmoms,
    #    automatic=True, truncate_by_symmetry=False,
    #    transformation_kwargs={"max_cell_size": largest_cell_size}).
    #    - automatic=True: pymatgen adds the ferrimagnetic and motif strategies
    #      where the structure has several magnetic species or Wyckoff positions.
    #    - truncate_by_symmetry=False: low-symmetry orderings often separate J's
    #      that symmetric ones cannot.
    #    - One call over all cell sizes up to largest_cell_size, so a strategy
    #      without orderings in the smallest cell (AFM with one magnetic site)
    #      finds them in larger ones. pymatgen keeps at most 64 orderings per
    #      strategy, the smaller cells first.
    # 5. rows = selection_rows(structure, enumerator.ordered_structures,
    #    heisenberg_settings); if their rank is below the number of columns, raise
    #    ValueError: the candidates can fix only that many of the parameters (E0
    #    and the J's), so the loop cannot converge and would only spend statics;
    #    lower the cutoff or raise max_magnetic_sites.
    # 6. Return enumerator.ordered_structures and ordered_structure_origins.
    raise NotImplementedError


def count_magnetic_sites(
    structure: Structure, default_magmoms: dict[str, float] | None
) -> int:
    """Return how many primitive-cell sites carry a moment under `default_magmoms`.

    The primitive cell, because the parent the orderings are enumerated on is one.
    """
    # 1. analyzer = CollinearMagneticStructureAnalyzer(structure,
    #    overwrite_magmom_mode="replace_all", default_magmoms=default_magmoms);
    #    by keyword, as the second positional argument is overwrite_magmom_mode.
    #    The analyzer reduces `structure` to its primitive cell by default.
    # 2. Return analyzer.number_of_magnetic_sites.
    raise NotImplementedError


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
    # 1. If `calculations` is not empty:
    #    a. structures, energies = collect_orderings(calculations)
    #    b. mapper = HeisenbergMapper(structures, energies, structure,
    #       **heisenberg_settings); not caught: pymatgen raises ValueError only for
    #       fewer than 2 distinct orderings or no J at all, and the first round
    #       selects at least as many distinct orderings as parameters (the prior
    #       makes those worth far more than any atom cost).
    #    c. model = mapper.get_heisenberg_model(); current = ExchangeRound(atoms of
    #       each ordering the last round added, len(structures), model.ex_params,
    #       model.residual, model.leave_one_out_error).
    #    d. every_target_resolved = set(mapper.dists) <= set(mapper.ex_mat.columns);
    #       pymatgen drops a J the orderings cannot separate from the shorter
    #       ones with only a warning, and the fit of the rest can still meet the
    #       tolerance.
    #    e. If every_target_resolved and current.leave_one_out_error
    #       <= leave_one_out_tolerance (meV per magnetic ion; pymatgen gives
    #       infinity while the orderings cannot fix every J): return
    #       ExchangeFit(model, rounds + [current], stop_reason="converged").
    #       Idea, untested: a fixed meV threshold passes too easily for weak magnets
    #       whose orderings lie within a few meV; dividing the error by the spread
    #       of the ordering energies would scale it to the material.
    # 2. selected, remaining = select_orderings(structure, candidates, structures
    #    (none before the first round), heisenberg_settings)
    # 3. If nothing is selected (only after a fit: enumerate_candidates already
    #    raised on a pool too small for the first round):
    #    a. stop_reason = "candidates_exhausted" if no candidates are left, else
    #       "not_worth_its_atoms".
    #    b. Warn that the J's did not converge, naming stop_reason, the last
    #       leave-one-out error and leave_one_out_tolerance, and what to try: if the
    #       DFT noise is near the tolerance, raise leave_one_out_tolerance; if
    #       exchange beyond the cutoff may matter, rerun with a larger cutoff in
    #       heisenberg_settings; if the candidates ran out, raise max_magnetic_sites.
    #    c. Return ExchangeFit(model, rounds + [current], stop_reason).
    # Next round: this job replaces itself with the statics of `selected` and a new
    # round, so its output becomes the final ExchangeFit.
    # 4. statics_job = run_ordering_calculations(selected, static_maker,
    #    relax_maker=None)
    #    DECISION 4: commensurate k-point meshes per supercell plug in here, on the
    #    static inputs of each ordering.
    # 5. next_round_job = run_exchange_round(structure, static_maker,
    #    heisenberg_settings, leave_one_out_tolerance, candidates=remaining,
    #    calculations=calculations + [statics_job.output],
    #    rounds=rounds + [current]).
    # 6. Return Response(replace=Flow([statics_job, next_round_job],
    #    output=next_round_job.output)).
    raise NotImplementedError


# --- Selecting informative orderings ---


def select_orderings(
    structure: Structure,
    candidates: Orderings,
    known_structures: list[Structure],
    heisenberg_settings: dict,
) -> tuple[Orderings, Orderings]:
    """Pick the candidates that add the most information per atom to the fit.

    Returns the selected and the remaining candidates.
    """
    # 1. known_rows = selection_rows(structure, known_structures,
    #    heisenberg_settings); candidate_rows = selection_rows(structure,
    #    candidates' structures, heisenberg_settings).
    # 2. information = known_rows.T @ known_rows + SELECTION_PRIOR * identity.
    # 3. Up to ORDERINGS_PER_PARAMETER * the number of parameters (columns of the
    #    rows) times:
    #    a. Score each candidate row x: log(1 + x @ inverse(information) @ x), its
    #       gain in log det(X^T X), minus ATOM_COST_WEIGHT * its atoms in parent
    #       cells' worth (len(candidate) / len(structure)).
    #    b. Stop if the best score is <= 0: no candidate is worth its atoms.
    #    c. Move the best candidate from the remaining to the selected;
    #       information += outer(x, x).
    # 4. Return the selected and the remaining candidates, each with its origins.
    raise NotImplementedError


def selection_rows(
    structure: Structure, orderings: list[Structure], heisenberg_settings: dict
) -> np.ndarray:
    """Return one design-matrix row per ordering: 1 for E0, then its spin signs per J.

    Every J the settings target gets a column, also those no ordering resolves yet.
    """
    # 1. parent = ParentCell(structure, magnetic species, cutoff and symprec from
    #    heisenberg_settings) (open: ParentCell and ComputedOrdering are not in
    #    pymatgen's documented API; ask to make them public?).
    # 2. For each ordering: ComputedOrdering(ordering, energy=0.0, parent=parent,
    #    matcher=...).spin_products(signs_only=True); the energy is unused, as
    #    only the spin products are read, and signs, because the moments are
    #    unknown before the static.
    # 3. Return the rows with E0 = 1 first and one column per J, shortest first, zero
    #    where an ordering has no bond of that J.
    raise NotImplementedError


# --- Fitting the orderings ---


def collect_orderings(
    calculations: list[RoundCalculations],
) -> tuple[list[Structure], list[float]]:
    """Return the structures with their output moments and their total energies."""
    # 1. For each task of each round: ordering = MagneticOrderingOutput
    #    .from_task_document(task), the schema for a static ordering calculation.
    # 2. Copy ordering.structure and add ordering.magmoms, the output moments, as
    #    the "magmom" site property; a static that flipped spins or lost moment is
    #    then fitted as the state it actually converged to.
    # 3. Return the structures and the ordering.energy values.
    raise NotImplementedError


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
