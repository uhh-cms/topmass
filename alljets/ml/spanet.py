# coding: utf-8

"""
Test model definition.
"""

from __future__ import annotations

import logging
from typing import Any

import law
import order as od

from columnflow.categorization import Categorizer, categorizer
from columnflow.config_util import add_category
from columnflow.ml import MLModel
from columnflow.util import maybe_import, dev_sandbox
from columnflow.columnar_util import Route, set_ak_column
from law.target.file import get_path
from columnflow.columnar_util import attach_coffea_behavior
from columnflow.production.util import delta_r_match

ak = maybe_import("awkward")
h5py = maybe_import("h5py")
np = maybe_import("numpy")
onnxruntime = maybe_import("onnxruntime")
prediction_selection = maybe_import("spanet.network.prediction_selection")
extract_predictions = getattr(prediction_selection, "extract_predictions", None)

logger = law.logger.get_logger(__name__)

# cut on the SPANet top assignment probabilities prob1, prob2, used by the categories below
SPANET_PROB_CUT = 0.2
# minimum delta R between the two SPANet b jets, used by the categories below
SPANET_DRBB_CUT = 2.0
# categories for events in 2btj with delta R_bb above the cut and both (sig) or exactly one (one) of prob1, prob2
# above the cut (disjoint)
SPANET_PROB_SIG_CAT_ID = 801
SPANET_PROB_ONE_CAT_ID = 802
# W mass used to scale the light jets of the constrained top candidates
SPANET_W_MASS = 80.4


def _spanet_2btj(events: ak.Array, config_inst: od.Config, cls_name: str) -> np.ndarray | None:
    # 2btj events with SPANet delta R_bb above the cut
    # SPANet columns only exist after the ML evaluation, so no event is selected before that
    if cls_name not in events.fields or "category_ids" not in events.fields:
        return None
    in_2btj = ak.to_numpy(ak.any(events.category_ids == config_inst.get_category("2btj").id, axis=1))
    return in_2btj & ak.to_numpy(events[cls_name].dRbb > SPANET_DRBB_CUT)


def spanet_prob_sig_mask(events: ak.Array, config_inst: od.Config, cls_name: str = "spanet") -> np.ndarray:
    in_2btj = _spanet_2btj(events, config_inst, cls_name)
    if in_2btj is None:
        return np.zeros(len(events), dtype=bool)
    sel1 = ak.to_numpy(events[cls_name].prob1 > SPANET_PROB_CUT)
    sel2 = ak.to_numpy(events[cls_name].prob2 > SPANET_PROB_CUT)
    return in_2btj & sel1 & sel2


def spanet_prob_one_mask(events: ak.Array, config_inst: od.Config, cls_name: str = "spanet") -> np.ndarray:
    in_2btj = _spanet_2btj(events, config_inst, cls_name)
    if in_2btj is None:
        return np.zeros(len(events), dtype=bool)
    sel1 = ak.to_numpy(events[cls_name].prob1 > SPANET_PROB_CUT)
    sel2 = ak.to_numpy(events[cls_name].prob2 > SPANET_PROB_CUT)
    return in_2btj & (sel1 ^ sel2)


@categorizer(uses=set())
def cat_spanet_prob_sig(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    2btj events with SPANet delta R_bb > 2 and both prob1 and prob2 > 0.2. The category id is assigned in
    SpaNetModel.evaluate, in the producers stage this categorizer selects no events since the SPANet columns don't
    exist yet.
    """
    return events, spanet_prob_sig_mask(events, self.config_inst)


@categorizer(uses=set())
def cat_spanet_prob_one(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    2btj events with SPANet delta R_bb > 2 and exactly one of prob1, prob2 > 0.2. As for cat_spanet_prob_sig, the
    category id is assigned in SpaNetModel.evaluate.
    """
    return events, spanet_prob_one_mask(events, self.config_inst)


class SpaNetModel(MLModel):
    def __init__(self, *args, folds: int | None = None, **kwargs, ):
        # mark the model as accepting only a single config
        single_config = True

        super().__init__(*args, **kwargs)

        # class- to instance-level attributes
        # (before being set, self.folds refers to a class-level attribute)
        self.folds = folds or self.folds
        self.max_njets = int(self.parameters.get("max_njets", 10))
        self.onnx_file = self.parameters.get(
            "onnx_file",
            "/afs/desy.de/user/s/stadie/xxl-af-cms/alljets_run2/spanet.onnx",
        )
        self.matches = ["b1", "w1q1", "w1q2", "b2", "w2q1", "w2q2"]
        self.probs = ["prob1", "prob2", "complete", "partial"]
        self.combtypes = ["combtype1", "combtype2", "combtype"]
        self.extras = ["dRbb"]
        self.recos = ["W1", "W2", "Top1", "Top2", "ConstrainedTop1", "ConstrainedTop2"]
        self.reco_fields = ["pt", "eta", "phi", "mass"]

    def output_names(self) -> list[str]:
        reco_names = [f"{reco}.{field}" for reco in self.recos for field in self.reco_fields]
        return self.matches + self.probs + self.combtypes + self.extras + reco_names

    def setup(self):
        # dynamically add variables for the quantities produced by this model
        if self.config_inst.has_tag(f"{self.cls_name}_called"):
            # call this function only once per config
            return

        for name in self.matches:
            self.config_inst.add_variable(
                name=f"{self.cls_name}.{name}",
                null_value=-1,
                binning=(20, -1.0, self.max_njets),
                x_title=f"{self.cls_name} {name}",
            )
        for name in self.probs:
            self.config_inst.add_variable(
                name=f"{self.cls_name}.{name}",
                null_value=0,
                binning=(20, 0, 1),
                x_title=f"{self.cls_name} {name}",
            )
        cls_name = self.cls_name
        self.config_inst.add_variable(
            name=f"{cls_name}.prob",
            expression=lambda events: events[cls_name].prob1 * events[cls_name].prob2,
            aux={"inputs": {f"{cls_name}.prob1", f"{cls_name}.prob2"}},
            null_value=0,
            binning=(20, 0, 1),
            x_title=f"{self.cls_name} prob",
        )
        self.config_inst.add_variable(
            name=f"{self.cls_name}.dRbb",
            binning=(50, 0, 5),
            x_title=rf"{self.cls_name} $\Delta R_{{bb}}$",
        )
        for name in self.combtypes:
            self.config_inst.add_variable(
                name=f"{self.cls_name}.{name}",
                null_value=0,
                binning=(4, -1.5, 2.5),
                x_title=f"{self.cls_name} {name}: 0: unmatched, 1: wrong, 2: correct",
            )
        # better of the two combination types
        self.config_inst.add_variable(
            name=f"{cls_name}.combtype_max",
            expression=lambda events: np.maximum(
                np.asarray(events[cls_name].combtype1), np.asarray(events[cls_name].combtype2),
            ),
            aux={"inputs": {f"{cls_name}.combtype1", f"{cls_name}.combtype2"}},
            null_value=0,
            binning=(4, -1.5, 2.5),
            x_title=f"{self.cls_name} max(combtype1, combtype2): 0: unmatched, 1: wrong, 2: correct",
        )
        # W and top candidates built from the SPANet jet assignments
        for reco in self.recos:
            mass_binning = (50, 60, 110) if reco.startswith("W") else (90, 50, 500)
            for field, binning, unit in [
                ("pt", (100, 0, 500), "GeV"),
                ("eta", (50, -5, 5), "1"),
                ("phi", (32, -3.2, 3.2), "1"),
                ("mass", mass_binning, "GeV"),
            ]:
                self.config_inst.add_variable(
                    name=f"{self.cls_name}.{reco}.{field}",
                    binning=binning,
                    unit=unit,
                    x_title=f"{self.cls_name} {reco} {field}",
                )
        # average masses of the two W and top candidates
        for name, reco, binning, label in [
            ("mtop_avg", "Top", (90, 50, 500), "top"),
            ("mw_avg", "W", (50, 60, 110), "W"),
            ("mtopc_avg", "ConstrainedTop", (90, 50, 500), "constrained top"),
        ]:
            self.config_inst.add_variable(
                name=f"{cls_name}.{name}",
                expression=lambda events, reco=reco: 0.5 * (
                    events[cls_name][f"{reco}1"].mass + events[cls_name][f"{reco}2"].mass
                ),
                aux={"inputs": {f"{cls_name}.{reco}1.mass", f"{cls_name}.{reco}2.mass"}},
                binning=binning,
                unit="GeV",
                x_title=f"{self.cls_name} average {label} mass",
            )

        # mass of the (constrained) top candidate with the higher assignment probability
        def mtop_best(events, reco):
            spanet = events[cls_name]
            return np.where(
                np.asarray(spanet.prob1) >= np.asarray(spanet.prob2),
                np.asarray(spanet[f"{reco}1"].mass),
                np.asarray(spanet[f"{reco}2"].mass),
            )

        for name, reco, label in [
            ("mtop_best", "Top", "top"),
            ("mtopc_best", "ConstrainedTop", "constrained top"),
        ]:
            self.config_inst.add_variable(
                name=f"{cls_name}.{name}",
                expression=lambda events, reco=reco: mtop_best(events, reco),
                aux={"inputs": {
                    f"{cls_name}.prob1", f"{cls_name}.prob2", f"{cls_name}.{reco}1.mass", f"{cls_name}.{reco}2.mass",
                }},
                null_value=-1,
                binning=(90, 50, 500),
                unit="GeV",
                x_title=f"{self.cls_name} {label} mass, highest prob",
            )
        if not self.config_inst.has_category("spanet_prob_sig"):
            add_category(
                self.config_inst,
                name="spanet_prob_sig",
                selection="cat_spanet_prob_sig",
                id=SPANET_PROB_SIG_CAT_ID,
                label=(
                    rf"$\geq$ 2 b-tagged jets, $\Delta R_{{bb}}$ > {SPANET_DRBB_CUT}, "
                    rf"both SPANet top probs > {SPANET_PROB_CUT}"
                ),
                tags={"2btj", "spanet"},
            )
        if not self.config_inst.has_category("spanet_prob_one"):
            add_category(
                self.config_inst,
                name="spanet_prob_one",
                selection="cat_spanet_prob_one",
                id=SPANET_PROB_ONE_CAT_ID,
                label=(
                    rf"$\geq$ 2 b-tagged jets, $\Delta R_{{bb}}$ > {SPANET_DRBB_CUT}, "
                    rf"one SPANet top prob > {SPANET_PROB_CUT}"
                ),
                tags={"2btj", "spanet"},
            )
        self.config_inst.add_tag(f"{self.cls_name}_called")

    def sandbox(self, task: law.Task) -> str:
        return dev_sandbox("bash::$AJ_BASE/sandboxes/spanet.sh")

    def datasets(self, config_inst: od.Config) -> set[od.Dataset]:
        return {
            config_inst.get_dataset("tt_fh_powheg"),
        }

    def uses(self, config_inst: od.Config) -> set[Route | str]:
        return {
            "SelectedJets.mass", "SelectedJets.pt", "SelectedJets.eta", "SelectedJets.phi",
            "SelectedJets.btagDeepFlavB", "gen_top.b.eta", "gen_top.b.phi", "gen_top.b.pt", "gen_top.b.mass",
            "gen_top.w_children.eta", "gen_top.w_children.phi", "gen_top.w_children.pt", "gen_top.w_children.mass",
            "category_ids",
        }

    def produces(self, config_inst: od.Config) -> set[Route | str]:
        return {f"{self.cls_name}.{name}" for name in self.output_names()} | {"category_ids"}

    def output(self, task: law.Task) -> law.FileSystemDirectoryTarget:
        return task.target(f"input_f{task.branch}of{self.folds}.h5", dir=True)

    def open_model(self, target: law.FileSystemDirectoryTarget):
        session = onnxruntime.InferenceSession(
            self.onnx_file,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        return session

    def train(
            self,
            task: law.Task,
            input: dict[str, list[dict[str, law.FileSystemFileTarget]]],
            output: law.FileSystemDirectoryTarget,
    ) -> None:
        # Create h5 file
        output.makedirs()
        fout = h5py.File(get_path(output), "w")

        # Create datasets
        h5_jet_mask = fout.create_dataset("INPUTS/Jets/MASK", (1000, self.max_njets), dtype='?',
                                          maxshape=(None, self.max_njets), chunks=(1000, self.max_njets))
        h5_jet_eta = fout.create_dataset("INPUTS/Jets/eta", (1000, self.max_njets), dtype='f',
                                         maxshape=(None, self.max_njets), chunks=(1000, self.max_njets))
        h5_jet_phi = fout.create_dataset("INPUTS/Jets/phi", (1000, self.max_njets), dtype='f',
                                         maxshape=(None, self.max_njets), chunks=(1000, self.max_njets))
        h5_jet_pt = fout.create_dataset("INPUTS/Jets/pt", (1000, self.max_njets), dtype='f',
                                        maxshape=(None, self.max_njets), chunks=(1000, self.max_njets))
        h5_jet_mass = fout.create_dataset("INPUTS/Jets/mass", (1000, self.max_njets), dtype='f',
                                          maxshape=(None, self.max_njets), chunks=(1000, self.max_njets))
        h5_jet_px = fout.create_dataset("INPUTS/Jets/px", (1000, self.max_njets), dtype='f',
                                        maxshape=(None, self.max_njets), chunks=(1000, self.max_njets))
        h5_jet_py = fout.create_dataset("INPUTS/Jets/py", (1000, self.max_njets), dtype='f',
                                        maxshape=(None, self.max_njets), chunks=(1000, self.max_njets))
        h5_jet_pz = fout.create_dataset("INPUTS/Jets/pz", (1000, self.max_njets), dtype='f',
                                        maxshape=(None, self.max_njets), chunks=(1000, self.max_njets))
        h5_jet_energy = fout.create_dataset("INPUTS/Jets/energy", (1000, self.max_njets), dtype='f',
                                            maxshape=(None, self.max_njets), chunks=(1000, self.max_njets))
        h5_jet_btag = fout.create_dataset("INPUTS/Jets/btag", (1000, self.max_njets), dtype='f',
                                          maxshape=(None, self.max_njets), chunks=(1000, self.max_njets))
        h5_target_t1b = fout.create_dataset("TARGETS/t1/b", (1000,), dtype='i', maxshape=(None,), chunks=1000)
        h5_target_t1q1 = fout.create_dataset("TARGETS/t1/q1", (1000,), dtype='i', maxshape=(None,), chunks=1000)
        h5_target_t1q2 = fout.create_dataset("TARGETS/t1/q2", (1000,), dtype='i', maxshape=(None,), chunks=1000)
        h5_target_t2b = fout.create_dataset("TARGETS/t2/b", (1000,), dtype='i', maxshape=(None,), chunks=1000)
        h5_target_t2q1 = fout.create_dataset("TARGETS/t2/q1", (1000,), dtype='i', maxshape=(None,), chunks=1000)
        h5_target_t2q2 = fout.create_dataset("TARGETS/t2/q2", (1000,), dtype='i', maxshape=(None,), chunks=1000)
        h5_class_complete = fout.create_dataset("CLASSIFICATIONS/EVENT/complete", (1000,), dtype='int64',
                                                maxshape=(None,),
                                                chunks=1000)
        h5_class_partial = fout.create_dataset("CLASSIFICATIONS/EVENT/partial", (1000,), dtype='int64',
                                               maxshape=(None,),
                                               chunks=1000)

        datasets = [h5_jet_eta, h5_jet_phi, h5_jet_pt, h5_jet_mass, h5_jet_px, h5_jet_py, h5_jet_pz, h5_jet_energy,
                    h5_jet_btag, h5_jet_mask, h5_target_t1b, h5_target_t1q1, h5_target_t1q2, h5_target_t2b,
                    h5_target_t2q1, h5_target_t2q2, h5_class_complete, h5_class_partial]
        # fill datasets
        index = 0
        complete = 0
        t1_comp = 0
        t2_comp = 0

        for dataset, files in input["events"][self.config_inst.name].items():
            for inp in files:
                events = ak.from_parquet(get_path(inp["mlevents"]))
                matches = self.get_matches(events)

                jets = ak.pad_none(events.SelectedJets, self.max_njets, axis=1, clip=True)
                nentries = len(events)
                for ds in datasets:
                    ds.resize(index + nentries, axis=0)
                # Get jet variables
                h5_jet_eta[index: index + nentries,] = ak.fill_none(jets.eta, 0)
                h5_jet_phi[index: index + nentries,] = ak.fill_none(jets.phi, 0)
                h5_jet_pt[index: index + nentries,] = ak.fill_none(jets.pt, 0)
                h5_jet_mass[index: index + nentries,] = ak.fill_none(jets.mass, 0)
                h5_jet_px[index: index + nentries,] = ak.fill_none(jets.pt, 0) * np.cos(ak.fill_none(jets.phi, 0))
                h5_jet_py[index: index + nentries,] = ak.fill_none(jets.pt * np.sin(jets.phi), 0)
                h5_jet_pz[index: index + nentries,] = ak.fill_none(jets.pt * np.sinh(jets.eta), 0)
                h5_jet_energy[index: index + nentries,] = ak.fill_none(np.sqrt(jets.mass ** 2 + (
                        jets.pt * np.cosh(jets.eta)) ** 2), 0)  # E = sqrt(m^2 + p^2) for massive jets
                h5_jet_btag[index: index + nentries,] = ak.fill_none(jets.btagDeepFlavB, 0)
                h5_jet_mask[index: index + nentries,] = ak.to_numpy(~ak.is_none(jets.eta, axis=1))
                h5_target_t1b[index: index + nentries,] = matches[:, 0]
                h5_target_t1q1[index: index + nentries,] = matches[:, 1]
                h5_target_t1q2[index: index + nentries,] = matches[:, 2]
                h5_target_t2b[index: index + nentries,] = matches[:, 3]
                h5_target_t2q1[index: index + nentries,] = matches[:, 4]
                h5_target_t2q2[index: index + nentries,] = matches[:, 5]
                h5_class_complete[index: index + nentries,] = np.sum(matches >= 0, axis=1) == 6
                h5_class_partial[index: index + nentries,] = ((np.sum(matches[:, 0:3] >= 0, axis=1) == 3) |
                                                              (np.sum(matches[:, 3:6] >= 0, axis=1) == 3))
                index += nentries
                complete += np.sum(np.sum(matches >= 0, axis=1) == 6)
                t1_comp += np.sum(np.sum(matches[:, 0:3] >= 0, axis=1) == 3)
                t2_comp += np.sum(np.sum(matches[:, 3:6] >= 0, axis=1) == 3)
        logger.debug(
            f"Statistic:\n - all: {index}\n - complete: {complete}\n"
            f" - t1 complete: {t1_comp}\n - t2 complete: {t2_comp}",
        )

        fout.close()

    def get_matches(self, events, return_partons: bool = False) -> Any:
        events = attach_coffea_behavior(events, {
            "SelectedJets": {
                "type_name": "Jet",
                "check_attr": "metric_table",
                "skip_fields": "*Idx*G",
            }, })
        gen_top = attach_coffea_behavior(
            events.gen_top,
            collections={
                "b": {
                    "type_name": "GenParticle",
                    "check_attr": "metric_table",
                    "skip_fields": "*Idx*G",
                },
                "w_children": {
                    "type_name": "GenParticle",
                    "check_attr": "metric_table",
                    "skip_fields": "*Idx*G",
                },
            },
        )

        # Compute delta_eta and delta_phi between b quarks and jets
        matches = np.empty((len(events), 6), dtype=int)
        partons = [gen_top.b[:, 0], gen_top.w_children[:, 0, 0], gen_top.w_children[:, 0, 1],
                   gen_top.b[:, 1], gen_top.w_children[:, 1, 0], gen_top.w_children[:, 1, 1]]
        for i, p in enumerate(partons):
            best_match_idxs, _ = delta_r_match(p, events.SelectedJets[:, 0:self.max_njets], max_dr=0.4,
                                               as_index=True)
            matches[:, i] = ak.fill_none(best_match_idxs[:, 0], -1).to_numpy()

        for i in range(self.max_njets):
            has_val = matches == i
            sel_rows = np.sum(has_val, axis=1) > 1
            mask = sel_rows[:, np.newaxis] & has_val
            matches[mask] = -1
        if return_partons:
            return matches, partons
        return matches

    def debug_mass_outliers(
            self,
            t: str,
            res: np.ndarray,
            correct: np.ndarray,
            gen_idx: np.ndarray,
            matches: np.ndarray,
            partons: list[ak.Array],
            jets_t: ak.Array,
            W: ak.Array,
            Top: ak.Array,
            mtop: float = 172.5,
            max_dm: float = 50.0,
            max_events: int = 20,
    ) -> None:
        # log correctly assigned triplets whose top mass is far from mtop, with the matched jets next to their partons
        top_mass = ak.to_numpy(Top.mass)
        w_mass = ak.to_numpy(W.mass)
        outlier = correct & (np.abs(top_mass - mtop) > max_dm)
        n_correct = np.sum(correct)
        logger.debug(
            f"t{t} mass outliers: {np.sum(outlier)} of {n_correct} correct triplets have |m_top - {mtop}| > {max_dm} "
            f"({np.sum(outlier) / max(n_correct, 1):.1%})",
        )
        # plain numpy copies of the fields: vector behaviors (e.g. .phi, .delta_r) fail on single records
        jet_fields = {f: ak.to_numpy(jets_t[f]) for f in ("pt", "eta", "phi", "btagDeepFlavB")}
        parton_fields = [{f: ak.to_numpy(ak.fill_none(p[f], np.nan)) for f in ("pt", "eta", "phi")} for p in partons]
        roles = ["b", "q1", "q2"]
        for i in np.flatnonzero(outlier)[:max_events]:
            g = gen_idx[i]
            logger.debug(f"  event {i}: t{t} = gen top {g + 1}, m_top = {top_mass[i]:.1f}, m_W = {w_mass[i]:.1f}")
            for j, role in enumerate(roles):
                jet = {f: v[i, j] for f, v in jet_fields.items()}
                # parton of the same gen top matched to this jet (q1 and q2 may be swapped)
                k = 3 * g + int(np.flatnonzero(matches[i, 3 * g:3 * g + 3] == res[i, j])[0])
                p = {f: v[i] for f, v in parton_fields[k].items()}
                dphi = (jet["phi"] - p["phi"] + np.pi) % (2 * np.pi) - np.pi
                dr = np.hypot(jet["eta"] - p["eta"], dphi)
                logger.debug(
                    f"    {role}: jet {res[i, j]} pt={jet['pt']:.1f} eta={jet['eta']:.2f} phi={jet['phi']:.2f} "
                    f"btag={jet['btagDeepFlavB']:.2f} | parton {roles[k % 3]} pt={p['pt']:.1f} "
                    f"eta={p['eta']:.2f} phi={p['phi']:.2f} | dR={dr:.2f} "
                    f"pt_jet/pt_parton={jet['pt'] / p['pt']:.2f}",
                )

    def evaluate(
            self,
            task: law.Task,
            events: ak.Array,
            models: list[Any],
            fold_indices: ak.Array,
            events_used_in_training: bool = False,
    ) -> ak.Array:
        # extract_predictions uses numba parallel=True; the default GNU OpenMP threading layer aborts
        # on any later fork() ("fork() called from a process already using GNU OpenMP"), so use fork-safe tbb
        import numba
        numba.config.THREADING_LAYER = "tbb"

        events = attach_coffea_behavior(events, {
            "SelectedJets": {
                "type_name": "Jet",
                "check_attr": "metric_table",
                "skip_fields": "*Idx*G",
            }, })
        logger.info(f"{self.cls_name}: evaluating {len(events)} events")
        debug = logger.isEnabledFor(logging.DEBUG)
        jets = ak.pad_none(events.SelectedJets, self.max_njets, axis=1, clip=True)
        feat_list = [
            ak.fill_none(np.log(1 + jets.pt), 0),
            ak.fill_none(jets.eta, 0),
            ak.fill_none(jets.phi, 0),
            ak.fill_none(np.log(1 + jets.mass), 0),
            # ak.fill_none(jets.px, 0),
            # ak.fill_none(jets.py, 0),
            # ak.fill_none(jets.pz, 0),
            ak.fill_none(np.log(1 + np.sqrt(jets.mass ** 2 + (jets.pt * np.cosh(jets.eta)) ** 2)), 0),
            ak.fill_none(jets.btagDeepFlavB, 0),
        ]

        # stack into (batch, max_njets, n_features)
        jets_data = np.stack([ak.to_numpy(f).astype(np.float32) for f in feat_list], axis=-1)
        del feat_list  # free memory
        # mask: True where jet exists (same as saved h5 mask)
        jets_mask = ak.to_numpy(~ak.is_none(jets.eta, axis=1)).astype(np.bool_)

        outputs = models[0].run(None, {"Jets_data": jets_data, "Jets_mask": jets_mask})
        del jets_data, jets_mask  # free memory after inference
        # ensure numpy float32 and contiguous layout
        preds = [np.ascontiguousarray(o, dtype=np.float32) for o in outputs[0:2]]
        outputs = [o.astype(np.float32) for o in outputs[2:]]

        # extract_predictions returns one array per prediction: shape (batch, partons)
        results = [r for r in extract_predictions(preds)]
        del preds  # free memory

        # the six assigned jets (b1, w1q1, w1q2, b2, w2q1, w2q2) must be distinct; negative values mark unassigned
        # partons and are not compared
        jet_ids = np.sort(np.concatenate(results[0:2], axis=1), axis=1)
        dup = np.any((jet_ids[:, 1:] == jet_ids[:, :-1]) & (jet_ids[:, 1:] >= 0), axis=1)
        if np.any(dup):
            i = np.flatnonzero(dup)[0]
            raise AssertionError(
                f"{self.cls_name}: {np.sum(dup)} events with a jet used in more than one assignment, "
                f"e.g. event {i}: t1 = {results[0][i]}, t2 = {results[1][i]}",
            )
        class_complete = outputs[2]
        class_partial = outputs[3]

        combtype1 = np.zeros(len(events), dtype=int)
        combtype2 = np.zeros(len(events), dtype=int)
        combtype = np.zeros(len(events), dtype=int)

        is_tt_fh = task.dataset_inst.has_tag("tt") and "fh" in task.dataset_inst.name
        has_gen = is_tt_fh and "gen_top" in events.fields and events.gen_top.ndim > 1
        if has_gen:
            matches, partons = self.get_matches(events, return_partons=True)
            gent1 = matches[:, 0:3]
            gent2 = matches[:, 3:6]

            def n_shared(gent, res):
                # number of jets of the SPANet triplet that are matched to a parton of this gen top
                return np.sum(np.any((gent[:, :, np.newaxis] == res[:, np.newaxis, :]) & (gent >= 0)[:, :, np.newaxis],
                                     axis=1), axis=1)

            def check_triplet(res):
                # compare the SPANet triplet (b, q1, q2) with the gen top whose b is matched to the same jet, or, if
                # the b jet matches neither gen b, with the gen top sharing more jets with the triplet; the triplet is
                # unmatched only if that gen top is not fully matched, otherwise it is wrong or correct
                in_t1 = gent1[:, 0] == res[:, 0]
                in_t2 = gent2[:, 0] == res[:, 0]
                use_t1 = in_t1 | (~in_t2 & (n_shared(gent1, res) >= n_shared(gent2, res)))
                gen = np.where(use_t1[:, np.newaxis], gent1, gent2)
                unmatched = np.any(gen == -1, axis=1)
                correct = ~unmatched & (np.all(gen == res, axis=1) | np.all(gen[:, [0, 2, 1]] == res, axis=1))
                return unmatched, correct, np.where(use_t1, 0, 1)

            unmatched1, correct1, gen_idx1 = check_triplet(results[0])
            unmatched2, correct2, gen_idx2 = check_triplet(results[1])

            combtype1[:] = np.asarray(~unmatched1, dtype=int) + np.asarray(correct1, dtype=int)
            combtype2[:] = np.asarray(~unmatched2, dtype=int) + np.asarray(correct2, dtype=int)
            combtype[:] = np.minimum(combtype1, combtype2)
            if debug:
                cat = self.config_inst.get_category("2btj")
                mask = ak.any(events.category_ids == cat.id, axis=1)
                for i in range(min(30, len(events))):
                    if not mask[i]:
                        continue
                    logger.debug(f"{i} {gent1[i]} {gent2[i]} {results[0][i]} {unmatched1[i]} {correct1[i]}")
                    logger.debug(f"{i} {gent1[i]} {gent2[i]} {results[1][i]} {unmatched2[i]} {correct2[i]}")
                    logger.debug(f"{i} {combtype1[i]} {combtype2[i]} {combtype[i]}")
                    logger.debug(f"{i} {outputs[0][i]} {outputs[1][i]} {class_complete[i]} {class_partial[i]}")
                gent1 = gent1[mask]
                gent2 = gent2[mask]
                unmatched1 = unmatched1[mask]
                unmatched2 = unmatched2[mask]
                correct1 = correct1[mask]
                correct2 = correct2[mask]
                matches = matches[mask]
                def _join(*values):
                    return " ".join(map(str, values))

                logger.debug(_join(
                    "t1:", len(gent1), np.sum(unmatched1), np.sum(correct1), np.sum(~unmatched1 & ~correct1),
                    np.sum(correct1) / np.sum(~unmatched1), np.mean(outputs[0][mask]),
                    np.mean(outputs[0][mask][correct1]),
                ))
                logger.debug(_join(
                    "t2:", len(gent2), np.sum(unmatched2), np.sum(correct2), np.sum(~unmatched2 & ~correct2),
                    np.sum(correct2) / np.sum(~unmatched2), np.mean(outputs[1][mask]),
                    np.mean(outputs[1][mask][correct2]),
                ))
                logger.debug(_join(
                    "comb.:", len(gent1), np.sum(unmatched1 | unmatched2), np.sum(correct1 & correct2),
                    np.sum(~(unmatched1 | unmatched2)) - np.sum(correct1 & correct2),
                    np.sum(correct1 & correct2) / np.sum(~(unmatched1 | unmatched2)),
                    np.mean(outputs[0][mask] * outputs[1][mask]),
                    np.mean(outputs[0][mask][correct1 & correct2] * outputs[1][mask][correct1 & correct2]),
                ))
                for cut in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    selected = (outputs[0][mask] * outputs[1][mask] > cut)
                    logger.debug(_join(
                        cut, np.sum(selected), np.sum((correct1 & correct2)[selected]),
                        np.sum(selected[~(unmatched1 | unmatched2)]),
                        np.sum((correct1 & correct2)[selected]) / np.sum(selected),
                        np.sum((correct1 & correct2)[selected]) / np.sum(selected[~(unmatched1 | unmatched2)]),
                        np.sum(selected[~(unmatched1 | unmatched2)]) / np.sum(~(unmatched1 | unmatched2)),
                        np.sum(selected[~(unmatched1 | unmatched2)]) / np.sum(selected),
                        np.sum((correct1 & correct2)[selected]) / np.sum(~(unmatched1 | unmatched2)),
                    ))
                match1 = np.sum(matches[:, 0:3] >= 0, axis=1) == 3
                match2 = np.sum(matches[:, 3:6] >= 0, axis=1) == 3

                logger.debug(_join(
                    "complete:", np.mean(class_complete[mask]), np.mean(class_complete[mask][(match1) & (match2)]),
                ))
                logger.debug(_join(
                    "partial:", np.mean(class_partial[mask]), np.mean(class_partial[mask][(match1) | (match2)]),
                    np.mean(class_partial[mask][(match1) ^ (match2)]), np.mean(class_partial[mask][(match1) & (match2)]),
                ))

        columns = {name: np.ascontiguousarray(results[i // 3][:, i % 3]) for i, name in enumerate(self.matches)}
        columns.update(
            prob1=outputs[0], prob2=outputs[1],
            complete=class_complete, partial=class_partial,
            combtype1=combtype1, combtype2=combtype2, combtype=combtype,
        )
        # W and top candidates from the assigned jets (b, q1, q2), as in kinFitMatch
        b_jets = []
        for t, res in zip(("1", "2"), results[0:2]):
            jets_t = events.SelectedJets[ak.from_regular(res, axis=1)]
            W = jets_t[:, 1].add(jets_t[:, 2])
            Top = jets_t[:, 0].add(W)
            # constrained top: light jets scaled so that their invariant mass is SPANET_W_MASS
            Top_c = jets_t[:, 0].add(W.multiply(SPANET_W_MASS / W.mass))
            b_jets.append(jets_t[:, 0])
            for field in self.reco_fields:
                columns[f"W{t}.{field}"] = getattr(W, field)
                columns[f"Top{t}.{field}"] = getattr(Top, field)
                columns[f"ConstrainedTop{t}.{field}"] = getattr(Top_c, field)
            if debug:
                for i in range(min(30, len(events))):
                    logger.debug(f"{i} {t} {combtype1[i]} {combtype2[i]} {res[i]} {W.mass[i]} {Top.mass[i]}")
            if debug and has_gen:
                correct, gen_idx = (correct1, gen_idx1) if t == "1" else (correct2, gen_idx2)
                self.debug_mass_outliers(t, res, correct, gen_idx, matches, partons, jets_t, W, Top)
        columns["dRbb"] = b_jets[0].delta_r(b_jets[1])
        assert set(columns) == set(self.output_names())
        for name, arr in columns.items():
            events = set_ak_column(events, f"{self.cls_name}.{name}", arr)

        # append the spanet_prob_sig / spanet_prob_one category ids (disjoint);
        # CreateHistograms lets this category_ids column overwrite the one from the producers
        new_ids = [events.category_ids]
        for mask_func, cat_id in [
            (spanet_prob_sig_mask, SPANET_PROB_SIG_CAT_ID),
            (spanet_prob_one_mask, SPANET_PROB_ONE_CAT_ID),
        ]:
            mask = mask_func(events, self.config_inst, self.cls_name)
            new_ids.append(ak.unflatten(np.full(np.sum(mask), cat_id, dtype=np.int64), mask.astype(np.int64)))
        events = set_ak_column(events, "category_ids", ak.concatenate(new_ids, axis=1), value_type=np.int64)

        logger.info(f"{self.cls_name}: done with {len(combtype)} events")
        return events


# usable derivations
example = SpaNetModel.derive("spanet", cls_dict={"folds": 1})
