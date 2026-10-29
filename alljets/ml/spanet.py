# coding: utf-8

"""
SPANet model for the assignment of jets to the partons of fully hadronic ttbar decays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import law
import order as od

from columnflow.categorization import Categorizer, categorizer
from columnflow.config_util import add_category
from columnflow.ml import MLModel
from columnflow.util import maybe_import, dev_sandbox
from columnflow.columnar_util import Route, set_ak_column, has_ak_column
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
# generated top mass of the nominal samples and of the top mass variations (shifts of tt_fh_powheg), written as
# INPUTS/Event/mtop_gen to the training input file for a top mass parametrized network
SPANET_MTOP_NOMINAL = 172.5
SPANET_MTOP_SHIFTS = {
    f"mtop{d}_{direction}": SPANET_MTOP_NOMINAL + sign * d
    for d in (1, 3, 6)
    for direction, sign in (("up", 1), ("down", -1))
}


def _attach_jet_behavior(events: ak.Array) -> ak.Array:
    return attach_coffea_behavior(events, {
        "SelectedJets": {
            "type_name": "Jet",
            "check_attr": "metric_table",
            "skip_fields": "*Idx*G",
        },
    })


def _spanet_prob_sels(
        events: ak.Array,
        config_inst: od.Config,
        cls_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    # 2btj events with SPANet delta R_bb above the cut, and the prob1, prob2 selections
    # SPANet columns only exist after the ML evaluation, so no event is selected before that
    if cls_name not in events.fields or "category_ids" not in events.fields:
        return None
    in_2btj = ak.to_numpy(ak.any(events.category_ids == config_inst.get_category("2btj").id, axis=1))
    in_2btj = in_2btj & ak.to_numpy(events[cls_name].dRbb > SPANET_DRBB_CUT)
    sel1 = ak.to_numpy(events[cls_name].prob1 > SPANET_PROB_CUT)
    sel2 = ak.to_numpy(events[cls_name].prob2 > SPANET_PROB_CUT)
    return in_2btj, sel1, sel2


def spanet_prob_sig_mask(events: ak.Array, config_inst: od.Config, cls_name: str = "spanet") -> np.ndarray:
    sels = _spanet_prob_sels(events, config_inst, cls_name)
    if sels is None:
        return np.zeros(len(events), dtype=bool)
    in_2btj, sel1, sel2 = sels
    return in_2btj & sel1 & sel2


def spanet_prob_one_mask(events: ak.Array, config_inst: od.Config, cls_name: str = "spanet") -> np.ndarray:
    sels = _spanet_prob_sels(events, config_inst, cls_name)
    if sels is None:
        return np.zeros(len(events), dtype=bool)
    in_2btj, sel1, sel2 = sels
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


# (name, id, mask function, label suffix) of the categories filled in SpaNetModel.evaluate
SPANET_CATEGORIES = [
    ("spanet_prob_sig", SPANET_PROB_SIG_CAT_ID, spanet_prob_sig_mask, "both SPANet top probs"),
    ("spanet_prob_one", SPANET_PROB_ONE_CAT_ID, spanet_prob_one_mask, "one SPANet top prob"),
]


def _triplet_complete(matches: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # events with all six partons, all three partons of top 1, and all three partons of top 2 matched to jets
    t1 = np.sum(matches[:, 0:3] >= 0, axis=1) == 3
    t2 = np.sum(matches[:, 3:6] >= 0, axis=1) == 3
    return t1 & t2, t1, t2


def _n_shared(gent: np.ndarray, res: np.ndarray) -> np.ndarray:
    # number of jets of the SPANet triplet that are matched to a parton of this gen top
    return np.sum(np.any((gent[:, :, np.newaxis] == res[:, np.newaxis, :]) & (gent >= 0)[:, :, np.newaxis],
                         axis=1), axis=1)


def _check_triplet(
        gent1: np.ndarray,
        gent2: np.ndarray,
        res: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # compare the SPANet triplet (b, q1, q2) with the gen top whose b is matched to the same jet, or, if
    # the b jet matches neither gen b, with the gen top sharing more jets with the triplet; the triplet is
    # unmatched only if that gen top is not fully matched, otherwise it is wrong or correct
    in_t1 = gent1[:, 0] == res[:, 0]
    in_t2 = gent2[:, 0] == res[:, 0]
    use_t1 = in_t1 | (~in_t2 & (_n_shared(gent1, res) >= _n_shared(gent2, res)))
    gen = np.where(use_t1[:, np.newaxis], gent1, gent2)
    unmatched = np.any(gen == -1, axis=1)
    correct = ~unmatched & (np.all(gen == res, axis=1) | np.all(gen[:, [0, 2, 1]] == res, axis=1))
    return unmatched, correct, np.where(use_t1, 0, 1)


@dataclass
class _GenComparison:
    # comparison of the SPANet triplets with the gen-matched jets, per triplet t1, t2
    matches: np.ndarray
    partons: list[ak.Array]
    unmatched: tuple[np.ndarray, np.ndarray]
    correct: tuple[np.ndarray, np.ndarray]
    gen_idx: tuple[np.ndarray, np.ndarray]


class SpaNetModel(MLModel):
    # h5 jet inputs written in train, as in the SPANet event file ttbar.yaml
    JET_FEATURES = ["eta", "phi", "pt", "mass", "energy", "btag"]
    # h5 targets, in the order of the columns of get_matches
    TARGETS = ["t1/b", "t1/q1", "t1/q2", "t2/b", "t2/q1", "t2/q2"]
    # h5 chunk size
    CHUNK = 1000

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
        self.probs = ["prob1", "prob2"]
        self.combtypes = ["combtype1", "combtype2", "combtype"]
        self.extras = ["dRbb"]
        self.recos = ["W1", "W2", "Top1", "Top2", "ConstrainedTop1", "ConstrainedTop2"]
        self.reco_fields = ["pt", "eta", "phi", "mass"]

    def parameter_pairs(self, only_significant: bool = False) -> list[tuple[str, Any]]:
        # the onnx file only selects the network used in evaluate, so it must not change the output paths
        pairs = super().parameter_pairs(only_significant=only_significant)
        if only_significant:
            pairs = [(name, value) for name, value in pairs if name != "onnx_file"]
        return pairs

    @property
    def parameters_repr(self) -> str:
        # no hash at all if only insignificant parameters are set, so that the paths match a run without settings
        if not self.parameter_pairs(only_significant=True):
            return ""
        return super().parameters_repr

    def output_names(self) -> list[str]:
        reco_names = [f"{reco}.{field}" for reco in self.recos for field in self.reco_fields]
        return self.matches + self.probs + self.combtypes + self.extras + reco_names

    def setup(self):
        # dynamically add variables and categories for the quantities produced by this model
        if self.config_inst.has_tag(f"{self.cls_name}_called"):
            # call this function only once per config
            return

        self._add_output_variables()
        self._add_derived_variables()
        for name, cat_id, _, label in SPANET_CATEGORIES:
            if self.config_inst.has_category(name):
                continue
            add_category(
                self.config_inst,
                name=name,
                selection=f"cat_{name}",
                id=cat_id,
                label=(
                    rf"$\geq$ 2 b-tagged jets, $\Delta R_{{bb}}$ > {SPANET_DRBB_CUT}, "
                    rf"{label} > {SPANET_PROB_CUT}"
                ),
                tags={"2btj", "spanet"},
            )
        self.config_inst.add_tag(f"{self.cls_name}_called")

    def _add_output_variables(self) -> None:
        # variables for the columns written in evaluate
        cls_name = self.cls_name
        for name in self.matches:
            self.config_inst.add_variable(
                name=f"{cls_name}_{name}",
                expression=f"{cls_name}.{name}",
                null_value=-1,
                binning=(20, -1.0, self.max_njets),
                x_title=f"{cls_name} {name}",
            )
        for name in self.probs:
            self.config_inst.add_variable(
                name=f"{cls_name}_{name}",
                expression=f"{cls_name}.{name}",
                null_value=0,
                binning=(20, 0, 1),
                x_title=f"{cls_name} {name}",
            )
        self.config_inst.add_variable(
            name=f"{cls_name}_dRbb",
            expression=f"{cls_name}.dRbb",
            binning=(50, 0, 5),
            x_title=rf"{cls_name} $\Delta R_{{bb}}$",
        )
        for name in self.combtypes:
            self.config_inst.add_variable(
                name=f"{cls_name}_{name}",
                expression=f"{cls_name}.{name}",
                null_value=0,
                binning=(4, -1.5, 2.5),
                x_title=f"{cls_name} {name}: 0: unmatched, 1: wrong, 2: correct",
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
                    name=f"{cls_name}_{reco}_{field}",
                    expression=f"{cls_name}.{reco}.{field}",
                    binning=binning,
                    unit=unit,
                    x_title=f"{cls_name} {reco} {field}",
                )

    def _add_derived_variables(self) -> None:
        # variables computed from the columns written in evaluate
        cls_name = self.cls_name
        self.config_inst.add_variable(
            name=f"{cls_name}_prob",
            expression=lambda events: events[cls_name].prob1 * events[cls_name].prob2,
            aux={"inputs": {f"{cls_name}.prob1", f"{cls_name}.prob2"}},
            null_value=0,
            binning=(20, 0, 1),
            x_title=f"{cls_name} prob",
        )
        # higher of the two top assignment probabilities
        self.config_inst.add_variable(
            name=f"{cls_name}_prob_max",
            expression=lambda events: np.maximum(
                np.asarray(events[cls_name].prob1), np.asarray(events[cls_name].prob2),
            ),
            aux={"inputs": {f"{cls_name}.prob1", f"{cls_name}.prob2"}},
            null_value=0,
            binning=(20, 0, 1),
            x_title=f"{cls_name} max(prob1, prob2)",
        )
        # better of the two combination types
        self.config_inst.add_variable(
            name=f"{cls_name}_combtype_max",
            expression=lambda events: np.maximum(
                np.asarray(events[cls_name].combtype1), np.asarray(events[cls_name].combtype2),
            ),
            aux={"inputs": {f"{cls_name}.combtype1", f"{cls_name}.combtype2"}},
            null_value=0,
            binning=(4, -1.5, 2.5),
            x_title=f"{cls_name} max(combtype1, combtype2): 0: unmatched, 1: wrong, 2: correct",
        )
        # average masses of the two W and top candidates
        for name, reco, binning, label in [
            ("mtop_avg", "Top", (90, 50, 500), "top"),
            ("mw_avg", "W", (50, 60, 110), "W"),
            ("mtopc_avg", "ConstrainedTop", (90, 50, 500), "constrained top"),
        ]:
            self.config_inst.add_variable(
                name=f"{cls_name}_{name}",
                expression=lambda events, reco=reco: 0.5 * (
                    events[cls_name][f"{reco}1"].mass + events[cls_name][f"{reco}2"].mass
                ),
                aux={"inputs": {f"{cls_name}.{reco}1.mass", f"{cls_name}.{reco}2.mass"}},
                binning=binning,
                unit="GeV",
                x_title=f"{cls_name} average {label} mass",
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
                name=f"{cls_name}_{name}",
                expression=lambda events, reco=reco: mtop_best(events, reco),
                aux={"inputs": {
                    f"{cls_name}.prob1", f"{cls_name}.prob2", f"{cls_name}.{reco}1.mass", f"{cls_name}.{reco}2.mass",
                }},
                null_value=-1,
                binning=(90, 50, 500),
                unit="GeV",
                x_title=f"{cls_name} {label} mass, highest prob",
            )

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
        } | {
            column
            for dataset_inst in self.datasets(config_inst)
            for column in self._event_weight_columns(config_inst, dataset_inst)
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

    def _jet_features(self, jets: ak.Array) -> dict[str, ak.Array]:
        # raw features of the padded jets (JET_FEATURES), missing jets are filled with 0
        wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
        features = {
            "eta": jets.eta,
            "phi": jets.phi,
            "pt": jets.pt,
            "mass": jets.mass,
            # E = sqrt(m^2 + p^2) for massive jets
            "energy": np.sqrt(jets.mass ** 2 + (jets.pt * np.cosh(jets.eta)) ** 2),
            # 1 for jets b-tagged at the tight working point, 0 otherwise
            "btag": ak.values_astype(jets.btagDeepFlavB >= wp_tight, np.float32),
        }
        return {name: ak.fill_none(features[name], 0) for name in self.JET_FEATURES}

    def _create_h5_datasets(self, fout: h5py.File) -> dict[str, h5py.Dataset]:
        # resizable datasets of the SPANet input file, filled in train
        def create(path, shape, dtype):
            return fout.create_dataset(
                path, (self.CHUNK,) + shape, dtype=dtype, maxshape=(None,) + shape, chunks=(self.CHUNK,) + shape,
            )

        jet_shape = (self.max_njets,)
        h5_datasets = {"INPUTS/Jets/MASK": create("INPUTS/Jets/MASK", jet_shape, "?")}
        for name in self.JET_FEATURES:
            h5_datasets[f"INPUTS/Jets/{name}"] = create(f"INPUTS/Jets/{name}", jet_shape, "f")
        for target in self.TARGETS:
            h5_datasets[f"TARGETS/{target}"] = create(f"TARGETS/{target}", (), "i")
        # global input, only used by SPANet if listed in the GLOBAL inputs of ttbar.yaml
        h5_datasets["INPUTS/Event/mtop_gen"] = create("INPUTS/Event/mtop_gen", (), "f")
        h5_datasets["INPUTS/Event/weight"] = create("INPUTS/Event/weight", (), "f")
        return h5_datasets

    def _event_weight_columns(self, config_inst: od.Config, dataset_inst: od.Dataset) -> list[str]:
        # weight columns from the event_weights aux entries of the config and the dataset, as in aj_hist
        if dataset_inst.is_data:
            return []
        return list(config_inst.x("event_weights", {})) + list(dataset_inst.x("event_weights", {}))

    def _event_weight(self, events: ak.Array, dataset_inst: od.Dataset) -> np.ndarray:
        # product of the nominal event weights, dataset-specific weights are skipped with a warning if missing
        weight = np.ones(len(events), dtype=np.float32)
        dataset_columns = set(dataset_inst.x("event_weights", {}))
        for column in self._event_weight_columns(self.config_inst, dataset_inst):
            if column in dataset_columns and not has_ak_column(events, column):
                logger.warning(f"weight '{column}' for dataset {dataset_inst.name} not found")
                continue
            weight = weight * ak.to_numpy(Route(column).apply(events)).astype(np.float32)
        return weight

    def training_shifts(self, config_inst: od.Config, dataset_inst: od.Dataset) -> set[str]:
        # top mass variations available for the dataset
        return {
            shift for shift in SPANET_MTOP_SHIFTS
            if config_inst.has_shift(shift) and shift in dataset_inst.info
        }

    def requires(self, task: law.Task) -> dict[str, dict[str, law.Task]]:
        # PrepareMLEvents workflows of the top mass variations, MLTraining only requires the nominal events
        from columnflow.tasks.ml import PrepareMLEvents

        return {
            dataset_inst.name: {
                shift: PrepareMLEvents.req(
                    task,
                    config=self.config_inst.name,
                    dataset=dataset_inst.name,
                    shift=shift,
                    branch=-1,
                    _exclude={"branches"},
                )
                for shift in sorted(self.training_shifts(self.config_inst, dataset_inst))
            }
            for dataset_inst in self.datasets(self.config_inst)
        }

    def _training_files(
            self,
            task: law.Task,
            input: dict[str, Any],
    ) -> list[tuple[law.FileSystemFileTarget, od.Dataset, float]]:
        # training event files with their dataset and generated top mass: the merged nominal events per fold and the
        # prepared events of the top mass variations of the same folds
        get_dataset = self.config_inst.get_dataset
        files = [
            (inp["mlevents"], get_dataset(dataset), SPANET_MTOP_NOMINAL)
            for dataset, dataset_files in input["events"][self.config_inst.name].items()
            for inp in dataset_files
        ]
        for dataset, shift_inputs in input["model"].items():
            for shift, inp in shift_inputs.items():
                for branch_output in inp["collection"].targets.values():
                    files.extend(
                        (branch_output["mlevents"][f], get_dataset(dataset), SPANET_MTOP_SHIFTS[shift])
                        for f in task.training_folds
                    )
        return files

    def train(
            self,
            task: law.Task,
            input: dict[str, list[dict[str, law.FileSystemFileTarget]]],
            output: law.FileSystemDirectoryTarget,
    ) -> None:
        # write the SPANet input file (h5), the training itself is done with SPANet
        output.makedirs()
        index = 0
        complete = 0
        t1_comp = 0
        t2_comp = 0
        with h5py.File(get_path(output), "w") as fout:
            h5_datasets = self._create_h5_datasets(fout)
            for target, dataset_inst, mtop_gen in self._training_files(task, input):
                events = ak.from_parquet(get_path(target))
                matches = self.get_matches(events)
                is_complete, is_t1, is_t2 = _triplet_complete(matches)
                nentries = len(events)

                jets = ak.pad_none(events.SelectedJets, self.max_njets, axis=1, clip=True)
                values = {"INPUTS/Jets/MASK": ak.to_numpy(~ak.is_none(jets.eta, axis=1))}
                for name, feature in self._jet_features(jets).items():
                    values[f"INPUTS/Jets/{name}"] = feature
                for i, name in enumerate(self.TARGETS):
                    values[f"TARGETS/{name}"] = matches[:, i]
                values["INPUTS/Event/mtop_gen"] = np.full(nentries, mtop_gen, dtype=np.float32)
                values["INPUTS/Event/weight"] = self._event_weight(events, dataset_inst)

                for path, ds in h5_datasets.items():
                    ds.resize(index + nentries, axis=0)
                    ds[index: index + nentries] = values[path]
                index += nentries
                complete += np.sum(is_complete)
                t1_comp += np.sum(is_t1)
                t2_comp += np.sum(is_t2)
        logger.debug(
            f"Statistic:\n - all: {index}\n - complete: {complete}\n"
            f" - t1 complete: {t1_comp}\n - t2 complete: {t2_comp}",
        )

    def get_matches(self, events, return_partons: bool = False) -> Any:
        events = _attach_jet_behavior(events)
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

    def _build_inputs(self, jets: ak.Array) -> tuple[np.ndarray, np.ndarray]:
        # SPANet inputs from the padded jets: features (batch, max_njets, n_features) and mask (batch, max_njets)
        # same features as in the h5 file, with the log_normalize ones of ttbar.yaml log-transformed
        features = self._jet_features(jets)
        feat_list = [
            np.log(1 + features["pt"]),
            features["eta"],
            features["phi"],
            np.log(1 + features["mass"]),
            np.log(1 + features["energy"]),
            features["btag"],
        ]
        jets_data = np.stack([ak.to_numpy(f).astype(np.float32) for f in feat_list], axis=-1)
        # mask: True where jet exists (same as saved h5 mask)
        jets_mask = ak.to_numpy(~ak.is_none(jets.eta, axis=1)).astype(np.bool_)
        return jets_data, jets_mask

    def _run_inference(
            self,
            session: Any,
            jets_data: np.ndarray,
            jets_mask: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        # returns the assigned jet triplets (one (batch, 3) array per top) and the outputs prob1, prob2
        outputs = session.run(None, {"Jets_data": jets_data, "Jets_mask": jets_mask})
        # ensure numpy float32 and contiguous layout
        preds = [np.ascontiguousarray(o, dtype=np.float32) for o in outputs[0:2]]
        outputs = [o.astype(np.float32) for o in outputs[2:]]
        # extract_predictions returns one array per prediction: shape (batch, partons)
        results = list(extract_predictions(preds))
        return results, outputs

    def _check_unique_assignments(self, results: list[np.ndarray]) -> None:
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

    def _classify_assignments(
            self,
            task: law.Task,
            events: ak.Array,
            results: list[np.ndarray],
    ) -> tuple[dict[str, np.ndarray], _GenComparison | None]:
        # combination types of the two SPANet triplets (0: unmatched, 1: wrong, 2: correct) and their minimum, all
        # zero without gen information
        combtypes = {name: np.zeros(len(events), dtype=int) for name in self.combtypes}
        is_tt_fh = task.dataset_inst.has_tag("tt") and "fh" in task.dataset_inst.name
        if not (is_tt_fh and "gen_top" in events.fields and events.gen_top.ndim > 1):
            return combtypes, None

        matches, partons = self.get_matches(events, return_partons=True)
        gent1 = matches[:, 0:3]
        gent2 = matches[:, 3:6]
        unmatched1, correct1, gen_idx1 = _check_triplet(gent1, gent2, results[0])
        unmatched2, correct2, gen_idx2 = _check_triplet(gent1, gent2, results[1])

        combtypes["combtype1"][:] = np.asarray(~unmatched1, dtype=int) + np.asarray(correct1, dtype=int)
        combtypes["combtype2"][:] = np.asarray(~unmatched2, dtype=int) + np.asarray(correct2, dtype=int)
        combtypes["combtype"][:] = np.minimum(combtypes["combtype1"], combtypes["combtype2"])
        gen = _GenComparison(
            matches=matches,
            partons=partons,
            unmatched=(unmatched1, unmatched2),
            correct=(correct1, correct2),
            gen_idx=(gen_idx1, gen_idx2),
        )
        return combtypes, gen

    def _log_assignment_stats(
            self,
            events: ak.Array,
            results: list[np.ndarray],
            outputs: list[np.ndarray],
            combtypes: dict[str, np.ndarray],
            gen: _GenComparison,
    ) -> None:
        # debug statistics of the SPANet assignments compared to the gen-matched jets for 2btj events
        prob1, prob2 = outputs
        combtype1, combtype2, combtype = (combtypes[name] for name in self.combtypes)
        gent1 = gen.matches[:, 0:3]
        gent2 = gen.matches[:, 3:6]
        unmatched1, unmatched2 = gen.unmatched
        correct1, correct2 = gen.correct

        cat = self.config_inst.get_category("2btj")
        mask = ak.any(events.category_ids == cat.id, axis=1)
        for i in range(min(30, len(events))):
            if not mask[i]:
                continue
            logger.debug(f"{i} {gent1[i]} {gent2[i]} {results[0][i]} {unmatched1[i]} {correct1[i]}")
            logger.debug(f"{i} {gent1[i]} {gent2[i]} {results[1][i]} {unmatched2[i]} {correct2[i]}")
            logger.debug(f"{i} {combtype1[i]} {combtype2[i]} {combtype[i]}")
            logger.debug(f"{i} {prob1[i]} {prob2[i]}")
        gent1 = gent1[mask]
        gent2 = gent2[mask]
        unmatched1 = unmatched1[mask]
        unmatched2 = unmatched2[mask]
        correct1 = correct1[mask]
        correct2 = correct2[mask]
        prob1 = prob1[mask]
        prob2 = prob2[mask]
        unmatched = unmatched1 | unmatched2
        correct = correct1 & correct2

        def _join(*values):
            return " ".join(map(str, values))

        logger.debug(_join(
            "t1:", len(gent1), np.sum(unmatched1), np.sum(correct1), np.sum(~unmatched1 & ~correct1),
            np.sum(correct1) / np.sum(~unmatched1), np.mean(prob1), np.mean(prob1[correct1]),
        ))
        logger.debug(_join(
            "t2:", len(gent2), np.sum(unmatched2), np.sum(correct2), np.sum(~unmatched2 & ~correct2),
            np.sum(correct2) / np.sum(~unmatched2), np.mean(prob2), np.mean(prob2[correct2]),
        ))
        logger.debug(_join(
            "comb.:", len(gent1), np.sum(unmatched), np.sum(correct),
            np.sum(~unmatched) - np.sum(correct),
            np.sum(correct) / np.sum(~unmatched),
            np.mean(prob1 * prob2),
            np.mean(prob1[correct] * prob2[correct]),
        ))
        for cut in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            selected = prob1 * prob2 > cut
            logger.debug(_join(
                cut, np.sum(selected), np.sum(correct[selected]),
                np.sum(selected[~unmatched]),
                np.sum(correct[selected]) / np.sum(selected),
                np.sum(correct[selected]) / np.sum(selected[~unmatched]),
                np.sum(selected[~unmatched]) / np.sum(~unmatched),
                np.sum(selected[~unmatched]) / np.sum(selected),
                np.sum(correct[selected]) / np.sum(~unmatched),
            ))

    def _build_candidates(
            self,
            events: ak.Array,
            results: list[np.ndarray],
            combtypes: dict[str, np.ndarray],
            gen: _GenComparison | None,
            debug: bool,
    ) -> dict[str, Any]:
        # W and top candidates from the assigned jets (b, q1, q2), as in kinFitMatch, and delta R of the two b jets
        columns = {}
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
                combtype1, combtype2 = combtypes["combtype1"], combtypes["combtype2"]
                for i in range(min(30, len(events))):
                    logger.debug(f"{i} {t} {combtype1[i]} {combtype2[i]} {res[i]} {W.mass[i]} {Top.mass[i]}")
            if debug and gen is not None:
                k = int(t) - 1
                self.debug_mass_outliers(
                    t, res, gen.correct[k], gen.gen_idx[k], gen.matches, gen.partons, jets_t, W, Top,
                )
        columns["dRbb"] = b_jets[0].delta_r(b_jets[1])
        return columns

    def _add_spanet_category_ids(self, events: ak.Array) -> ak.Array:
        # append the spanet_prob_sig / spanet_prob_one category ids (disjoint);
        # CreateHistograms lets this category_ids column overwrite the one from the producers
        new_ids = [events.category_ids]
        for _, cat_id, mask_func, _ in SPANET_CATEGORIES:
            mask = mask_func(events, self.config_inst, self.cls_name)
            new_ids.append(ak.unflatten(np.full(np.sum(mask), cat_id, dtype=np.int64), mask.astype(np.int64)))
        return set_ak_column(events, "category_ids", ak.concatenate(new_ids, axis=1), value_type=np.int64)

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

        events = _attach_jet_behavior(events)
        logger.info(f"{self.cls_name}: evaluating {len(events)} events")
        debug = logger.isEnabledFor(logging.DEBUG)

        jets = ak.pad_none(events.SelectedJets, self.max_njets, axis=1, clip=True)
        jets_data, jets_mask = self._build_inputs(jets)
        results, outputs = self._run_inference(models[0], jets_data, jets_mask)
        del jets_data, jets_mask  # free memory after inference
        self._check_unique_assignments(results)

        combtypes, gen = self._classify_assignments(task, events, results)
        if debug and gen is not None:
            self._log_assignment_stats(events, results, outputs, combtypes, gen)

        columns = {name: np.ascontiguousarray(results[i // 3][:, i % 3]) for i, name in enumerate(self.matches)}
        columns.update(zip(self.probs, outputs))
        columns.update(combtypes)
        columns.update(self._build_candidates(events, results, combtypes, gen, debug))
        assert set(columns) == set(self.output_names())
        for name, arr in columns.items():
            events = set_ak_column(events, f"{self.cls_name}.{name}", arr)

        events = self._add_spanet_category_ids(events)

        logger.info(f"{self.cls_name}: done with {len(events)} events")
        return events


# usable derivations
example = SpaNetModel.derive("spanet", cls_dict={"folds": 1})
