"""Tests for the Heisenberg exchange workflow.

The material is a toy MnO cell (CsCl type, Mn on a simple cubic lattice). A model
static maker stands in for DFT and gives each ordering its energy under known
exchange constants, so every converged fit must recover them.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose

import pytest
from emmet.core.tasks import OutputDoc, TaskDoc
from jobflow import Maker, job, run_locally
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate2.common.flows.exchange import ExchangeMaker
from atomate2.common.jobs import exchange as exchange_jobs
from atomate2.common.jobs.exchange import (
    MAX_CELL_SIZE,
    enumerate_candidates,
    run_exchange_round,
    select_orderings,
    symmetrize_to_paramagnetic_parent,
)
from atomate2.common.schemas.exchange import ExchangeDocument, ExchangeFit

LATTICE_CONSTANT = 4.0  # Angstrom
MOMENT = 3.0  # muB on each Mn
CUTOFF = 6.0  # Angstrom: couples the 4 and 5.66 Angstrom Mn-Mn bonds, not 6.93
E0 = -5.0  # eV per Mn
J1 = -2.0  # meV/muB^2, nearest Mn-Mn bond
J2 = 0.3  # meV/muB^2, second-nearest Mn-Mn bond
PRIM_CUBIC_SPACE_GROUP = 221  # Pm-3m
SETTINGS = {"cutoff": CUTOFF, "default_magmoms": {"Mn": MOMENT}}

# Per ordering: the supercell and the spin sign of the Mn at lattice vector n.
ORDERINGS = {
    "fm": ((1, 1, 1), lambda _n: 1),
    "a_afm": ((2, 1, 1), lambda n: (-1) ** n[0]),
    "c_afm": ((2, 2, 1), lambda n: (-1) ** (n[0] + n[1])),
    "g_afm": ((2, 2, 2), lambda n: (-1) ** sum(n)),
}


def make_ordering(parent: Structure, scaling, spin_sign) -> Structure:
    """Return the supercell `scaling` of `parent` with Mn moments `spin_sign`."""
    supercell = parent * scaling
    mn_sites = [site for site in supercell if site.specie.symbol == "Mn"]
    origin = mn_sites[0].coords
    magmoms = []
    for site in supercell:
        if site.specie.symbol == "Mn":
            lattice_vector = [
                round(x) for x in (site.coords - origin) / LATTICE_CONSTANT
            ]
            magmoms.append(MOMENT * spin_sign(lattice_vector))
        else:
            magmoms.append(0.0)
    supercell.add_site_property("magmom", magmoms)
    return supercell


def make_orderings(parent: Structure) -> tuple[list[Structure], list[str]]:
    structures = [make_ordering(parent, *ordering) for ordering in ORDERINGS.values()]
    return structures, list(ORDERINGS)


def model_energy(structure: Structure) -> float:
    """Total energy (eV) of `structure` under E0, J1 and J2, from its Mn moments."""
    moments = structure.site_properties["magmom"]
    is_mn = [site.specie.symbol == "Mn" for site in structure]
    energy = E0 * sum(is_mn)
    centers, neighbors, _images, distances = structure.get_neighbor_list(CUTOFF)
    for i, j, distance in zip(centers, neighbors, distances, strict=True):
        if is_mn[i] and is_mn[j]:
            exchange = J1 if isclose(distance, LATTICE_CONSTANT) else J2
            # Each bond is met from both ends.
            energy -= exchange * moments[i] * moments[j] / 1000 / 2
    return energy


@dataclass
class ModelStaticMaker(Maker):
    """Stands in for DFT: each ordering gets its model energy and keeps its moments.

    `ferromagnetic_shift` (eV per Mn) is an energy the Heisenberg model cannot fit.
    """

    name: str = "model static"
    ferromagnetic_shift: float = 0.0

    @job
    def make(self, structure: Structure) -> TaskDoc:
        energy = model_energy(structure)
        if all(moment >= 0 for moment in structure.site_properties["magmom"]):
            energy += self.ferromagnetic_shift * structure.composition["Mn"]
        return TaskDoc(
            structure=structure,
            output=OutputDoc(energy=energy, structure=structure),
            dir_name="model",
            calcs_reversed=[],
        )


class ModelEnumerator:
    """Stands in for MagneticStructureEnumerator, which needs enumlib."""

    def __init__(self, structure: Structure, **_kwargs):
        self.ordered_structures, self.ordered_structure_origins = make_orderings(
            structure
        )


@pytest.fixture
def parent() -> Structure:
    return Structure(
        Lattice.cubic(LATTICE_CONSTANT), ["Mn", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


@pytest.fixture
def model_enumerator(monkeypatch):
    monkeypatch.setattr(exchange_jobs, "MagneticStructureEnumerator", ModelEnumerator)


def assert_recovers_exchange(ex_params: dict) -> None:
    assert ex_params == pytest.approx(
        {"E0": E0, "J1(Mn1-Mn1)": J1, "J2(Mn1-Mn1)": J2}, abs=1e-6
    )


def test_symmetrize_removes_moments_and_distortion(parent):
    distorted = parent.copy()
    distorted.translate_sites([0], [0.01, 0, 0])
    distorted.add_site_property("magmom", [MOMENT, 0.0])

    symmetrized = symmetrize_to_paramagnetic_parent.original(distorted)

    assert "magmom" not in symmetrized.site_properties
    analyzer = SpacegroupAnalyzer(symmetrized)
    assert analyzer.get_space_group_number() == PRIM_CUBIC_SPACE_GROUP


def test_enumerate_candidates_rejects_too_few_orderings(parent, monkeypatch):
    class FerromagnetOnly(ModelEnumerator):
        def __init__(self, structure: Structure, **_kwargs):
            self.ordered_structures = [make_ordering(parent, *ORDERINGS["fm"])]
            self.ordered_structure_origins = ["fm"]

    monkeypatch.setattr(exchange_jobs, "MagneticStructureEnumerator", FerromagnetOnly)

    with pytest.raises(ValueError, match="can fix only 1 of the 3 parameters"):
        enumerate_candidates.original(parent, SETTINGS, max_magnetic_sites=8)


def test_select_orderings_prefers_the_smaller_supercell(parent):
    a_afm_scaling, a_afm_sign = ORDERINGS["a_afm"]
    candidates = (
        [
            make_ordering(parent, (4, 1, 1), a_afm_sign),
            make_ordering(parent, a_afm_scaling, a_afm_sign),
        ],
        ["a_afm_large", "a_afm_small"],
    )
    known = [make_ordering(parent, *ORDERINGS["fm"])]

    (_structures, origins), _remaining = select_orderings(
        parent, candidates, known, SETTINGS
    )

    assert origins[0] == "a_afm_small"


def test_select_orderings_skips_orderings_not_worth_their_atoms(parent):
    # Repeats make the fit so well determined that one more ordering gains little.
    known, _origins = make_orderings(parent)
    candidates = ([make_ordering(parent, *ORDERINGS["g_afm"])], ["g_afm"])

    selected, remaining = select_orderings(parent, candidates, known * 5, SETTINGS)

    assert selected == ([], [])
    assert remaining[1] == ["g_afm"]


def test_run_exchange_round_converges_on_exact_energies(parent, memory_jobstore):
    round_job = run_exchange_round(
        parent,
        ModelStaticMaker(),
        SETTINGS,
        leave_one_out_tolerance=1.0,
        candidates=make_orderings(parent),
        calculations=[],
        rounds=[],
    )

    run_locally(round_job, store=memory_jobstore, ensure_success=True)
    fit = round_job.output.resolve(memory_jobstore)

    assert isinstance(fit, ExchangeFit)
    assert fit.stop_reason == "converged"
    assert_recovers_exchange(fit.heisenberg_model.ex_params)
    assert len(fit.rounds) == 1
    assert sorted(fit.rounds[0].new_ordering_atoms) == [2, 4, 8, 16]


def test_run_exchange_round_stops_when_candidates_run_out(parent, memory_jobstore):
    round_job = run_exchange_round(
        parent,
        ModelStaticMaker(ferromagnetic_shift=0.01),
        SETTINGS,
        leave_one_out_tolerance=1.0,
        candidates=make_orderings(parent),
        calculations=[],
        rounds=[],
    )

    with pytest.warns(UserWarning, match="candidates_exhausted"):
        run_locally(round_job, store=memory_jobstore, ensure_success=True)
    fit = round_job.output.resolve(memory_jobstore)

    assert fit.stop_reason == "candidates_exhausted"
    assert fit.rounds[-1].leave_one_out_error > 1.0


def test_make_fits_exchange_on_the_parent(parent, model_enumerator):
    maker = ExchangeMaker(
        static_maker=ModelStaticMaker(),
        heisenberg_settings={"cutoff": CUTOFF},
        default_magmoms={"Mn": MOMENT},
    )
    flow = maker.make(parent)

    response = run_locally(flow, create_folders=False, ensure_success=True)
    doc = response[flow.output.uuid][1].output

    assert isinstance(doc, ExchangeDocument)
    assert_recovers_exchange(doc.ex_params)
    assert doc.stop_reason == "converged"
    assert doc.loop_limits["max_cell_size"] == MAX_CELL_SIZE
    assert doc.formula_pretty == "MnO"
    assert maker.heisenberg_settings == {"cutoff": CUTOFF}


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"leave_one_out_tolerance": 0}, "must be positive"),
        ({"heisenberg_settings": {"default_magmoms": {}}}, "not in heisenberg"),
        ({"default_magmoms": {"Fe": MOMENT}}, "no magnetic site"),
        ({"max_magnetic_sites": 0}, "more than max_magnetic_sites"),
    ],
)
def test_make_rejects_inputs_that_cannot_start_the_loop(parent, settings, message):
    with pytest.raises(ValueError, match=message):
        ExchangeMaker(**settings).make(parent)
