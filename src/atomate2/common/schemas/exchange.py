"""Schemas for magnetic exchange (Heisenberg + Vampire) calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field
from pymatgen.analysis.magnetism.heisenberg import HeisenbergModel
from pymatgen.core.structure import Structure

if TYPE_CHECKING:
    from atomate2.vampire.schemas.vampire_output import VampireOutput

# Why the convergence loop ended: the fit met the tolerance, every candidate was
# computed, or no candidate left was worth its atoms.
StopReason = Literal["converged", "candidates_exhausted", "not_worth_its_atoms"]

# JSON has no infinity; write `Infinity` so an infinite leave-one-out error reloads.
KEEP_INFINITY = ConfigDict(ser_json_inf_nan="constants")


class ExchangeRound(BaseModel):
    """Summary of the Heisenberg fit after one round of selected orderings."""

    model_config = KEEP_INFINITY

    new_ordering_atoms: list[int] = Field(
        description="Number of atoms in the supercell of each ordering the round added."
    )
    num_orderings: int = Field(
        description="Number of orderings the fit used, from this and earlier rounds."
    )
    ex_params: dict[str, float] = Field(
        description="Fitted J_ij in meV/muB^2 and 'E0' in eV per magnetic ion."
    )
    residual: float = Field(
        description="RMS fit residual in meV per magnetic ion."
    )
    leave_one_out_error: float = Field(
        description="RMS error of predicting each ordering's energy from a fit to all "
        "the others, in meV per magnetic ion; infinite if one ordering alone fixes a J."
    )


class ExchangeFit(BaseModel):
    """Final Heisenberg model of the convergence loop and the summary of each round."""

    model_config = KEEP_INFINITY

    heisenberg_model: HeisenbergModel = Field(
        description="The model fitted on the orderings of all rounds."
    )
    rounds: list[ExchangeRound] = Field(
        description="One summary per round, in the order the rounds ran."
    )
    stop_reason: StopReason = Field(description="Why the loop ended.")


class ExchangeDocument(BaseModel):
    """Final document with fitted Heisenberg exchange parameters and Tc.

    This is the output of the ExchangeMaker workflow. The Heisenberg fields are
    always populated; the Vampire fields are populated only if the Monte-Carlo
    critical-temperature step was run.
    """

    model_config = KEEP_INFINITY

    formula: str | None = Field(
        None,
        description="Formula taken from pymatgen.core.structure.Structure.formula.",
    )
    formula_pretty: str | None = Field(
        None,
        description="Cleaned representation of the formula.",
    )
    parent_structure: Structure | None = Field(
        None,
        description="The paramagnetic parent cell the orderings were mapped onto.",
    )
    heisenberg_settings: dict | None = Field(
        None,
        description="The {cutoff, symprec} the HeisenbergMapper fitted the model with.",
    )
    vampire_settings: dict | None = Field(
        None,
        description="Keyword arguments passed to the Vampire Monte-Carlo run (e.g. "
        "mc_box_size, equil_timesteps, mc_timesteps), if run. Unset keys fall "
        "back to the VampireCaller defaults.",
    )
    residual: float | None = Field(
        None,
        description="Root-mean-square residual of the least-squares fit that "
        "produced ex_params, in meV per magnetic ion. Only meaningful with many "
        "more orderings than fitted parameters; otherwise the fit overfits and the "
        "residual is low without the fit being good.",
    )
    leave_one_out_error: float | None = Field(
        None,
        description="RMS error of predicting each ordering's energy from a fit to all "
        "the others, in meV per magnetic ion; the measure the convergence loop stops "
        "on, and infinite if one ordering alone fixes a J.",
    )
    ex_params: dict | None = Field(
        None,
        description="Fitted exchange parameters keyed by interaction label. The "
        "J_ij are in meV/muB^2 (they multiply the raw moments); the included 'E0' "
        "offset is in eV per magnetic ion.",
    )
    ex_mat: dict | None = Field(
        None,
        description="Heisenberg Hamiltonian matrix used for the Heisenberg model fit.",
    )
    heisenberg_model: dict | None = Field(
        None, description="Full HeisenbergModel as a serialized dict (as_dict())."
    )
    rounds: list[ExchangeRound] | None = Field(
        None,
        description="Fit summary after each round of the convergence loop.",
    )
    stop_reason: StopReason | None = Field(
        None,
        description="Why the convergence loop ended; only 'converged' means the fit "
        "met leave_one_out_tolerance.",
    )
    loop_limits: dict[str, float] | None = Field(
        None,
        description="The limits of the convergence loop the fit ran with: the maker's "
        "leave_one_out_tolerance and max_magnetic_sites, and the fixed "
        "max_cell_size, orderings_per_parameter and atom_cost_weight.",
    )
    critical_temp: float | None = Field(
        None,
        description="Critical (Curie/Neel) temperature in Kelvin from Vampire, if run.",
    )
    vampire_output: dict | None = Field(
        None, description="Full VampireOutput as a serialized dict (as_dict()), if run."
    )

    @classmethod
    def from_model(
        cls,
        heisenberg_model: HeisenbergModel,
        parent_structure: Structure,
        rounds: list[ExchangeRound] | None = None,
        stop_reason: StopReason | None = None,
        loop_limits: dict[str, float] | None = None,
        vampire_output: VampireOutput | None = None,
        vampire_settings: dict | None = None,
    ) -> ExchangeDocument:
        """Construct an ExchangeDocument from a fitted model and optional Vampire run.

        Parameters
        ----------
        heisenberg_model : HeisenbergModel
            The fitted Heisenberg model from pymatgen's HeisenbergMapper.
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
            The keyword arguments the Vampire Monte-Carlo run was called with, if run.
        """
        return cls(
            formula=parent_structure.formula,
            formula_pretty=parent_structure.composition.reduced_formula,
            parent_structure=parent_structure,
            heisenberg_settings={
                "cutoff": heisenberg_model.cutoff,
                "symprec": heisenberg_model.symprec,
            },
            vampire_settings=vampire_settings if vampire_output else None,
            residual=heisenberg_model.residual,
            leave_one_out_error=heisenberg_model.leave_one_out_error,
            ex_params=heisenberg_model.ex_params,
            ex_mat=heisenberg_model.ex_mat.to_dict(),
            heisenberg_model=heisenberg_model.as_dict(),
            rounds=rounds,
            stop_reason=stop_reason,
            loop_limits=loop_limits,
            critical_temp=vampire_output.critical_temp if vampire_output else None,
            vampire_output=vampire_output.as_dict() if vampire_output else None,
        )
