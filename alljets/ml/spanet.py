# coding: utf-8

"""
Test model definition.
"""

from __future__ import annotations

from typing import Any

import law
import order as od

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

class SpaNetModel(MLModel):
    def __init__(self, *args, folds: int | None = None, **kwargs, ):
        # mark the model as accepting only a single config
        single_config = True

        super().__init__(*args, **kwargs)

        # class- to instance-level attributes
        # (before being set, self.folds refers to a class-level attribute)
        self.folds = folds or self.folds
        self.max_njets = 10
        self.matches = ["b1", "w1q1", "w1q2", "b2", "w2q1", "w2q2"]

    def setup(self):
        # dynamically add variables for the quantities produced by this model
        for name in self.matches:
            if f"{self.cls_name}.{name}" not in self.config_inst.variables:
                self.config_inst.add_variable(
                    name=f"{self.cls_name}.{name}",
                    null_value=-1,
                    binning=(20, -1.0, self.max_njets),
                    x_title=f"{self.cls_name} SPANet {name}",
                )
        for name in ["prob1", "prob2"]:
            if f"{self.cls_name}.{name}" not in self.config_inst.variables:
                self.config_inst.add_variable(
                    name=f"{self.cls_name}.{name}",
                    null_value=0,
                    binning=(20, 0, 1),
                    x_title=f"{self.cls_name} SPANet {name}",
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
        }

    def produces(self, config_inst: od.Config) -> set[Route | str]:
        return {
            f"{self.cls_name}.output",
        }

    def output(self, task: law.Task) -> law.FileSystemDirectoryTarget:
        return task.target(f"input_f{task.branch}of{self.folds}.h5", dir=True)

    def open_model(self, target: law.FileSystemDirectoryTarget):
        return None

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

        datasets = [h5_jet_eta, h5_jet_phi, h5_jet_pt, h5_jet_mass, h5_jet_px, h5_jet_py, h5_jet_pz, h5_jet_energy,
                    h5_jet_btag, h5_jet_mask, h5_target_t1b, h5_target_t1q1, h5_target_t1q2, h5_target_t2b,
                    h5_target_t2q1, h5_target_t2q2]
        # fill datasets
        index = 0
        complete = 0
        t1_comp = 0
        t2_comp = 0
        cat = self.config_inst.get_category("2btj")
        for dataset, files in input["events"][self.config_inst.name].items():
            for inp in files:
                events = ak.from_parquet(get_path(inp["mlevents"]))
                mask = ak.any(events.category_ids == cat.id, axis=1)
                events = events[mask]
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
                for i, p in enumerate([gen_top.b[:, 0], gen_top.w_children[:, 0, 0], gen_top.w_children[:, 0, 1],
                                       gen_top.b[:, 1], gen_top.w_children[:, 1, 0], gen_top.w_children[:, 1, 1]]):
                    best_match_idxs, _ = delta_r_match(p, events.SelectedJets[:, 0:self.max_njets], max_dr=0.4, as_index=True)
                    matches[:, i] = ak.fill_none(best_match_idxs[:, 0], -1).to_numpy()

                for i in range(self.max_njets):
                    has_val = matches == i
                    sel_rows = np.sum(has_val, axis=1) > 1
                    mask = sel_rows[:, np.newaxis] & has_val
                    matches[mask] = -1

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
                index += nentries
                complete += np.sum(np.sum(matches >= 0, axis=1) == 6)
                t1_comp += np.sum(np.sum(matches[:, 0:3] >= 0, axis=1) == 3)
                t2_comp += np.sum(np.sum(matches[:, 3:6] >= 0, axis=1) == 3)

        print("Statistic:")
        print(f" - all: {index}")
        print(f" - complete: {complete}")
        print(f" - t1 complete: {t1_comp}")
        print(f" - t2 complete: {t2_comp}")

        fout.close()

    def evaluate(
            self,
            task: law.Task,
            events: ak.Array,
            models: list[Any],
            fold_indices: ak.Array,
            events_used_in_training: bool = False,
    ) -> ak.Array:
        session = onnxruntime.InferenceSession(
            "./spanet.onnx",
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        jets = ak.pad_none(events.SelectedJets, self.max_njets, axis=1, clip=True)
        feat_list = [
            ak.fill_none(np.log(1 + jets.pt), 0),
            ak.fill_none(jets.eta, 0),
            ak.fill_none(jets.phi, 0),
            ak.fill_none(np.log(1 + np.sqrt(jets.mass ** 2 + (jets.pt * np.cosh(jets.eta)) ** 2)), 0),
            ak.fill_none(jets.btagDeepFlavB, 0),
        ]

        # stack into (batch, max_njets, n_features)
        jets_data = np.stack([ak.to_numpy(f).astype(np.float32) for f in feat_list], axis=-1)
        # mask: True where jet exists (same as saved h5 mask)
        jets_mask = ak.to_numpy(~ak.is_none(jets.eta, axis=1)).astype(np.bool_)

        outputs = session.run(None, {"Jets_data": jets_data, "Jets_mask": jets_mask})
        # ensure numpy float32 and contiguous layout
        preds = [np.ascontiguousarray(o.astype(np.float32)) for o in outputs[0:2]]

        # extract_predictions returns one array per prediction: shape (batch, partons)
        results = extract_predictions(preds)

        for i, name in enumerate(self.matches):
            events = set_ak_column(events, f"{self.cls_name}.{name}", results[i//3][:,i%3])
        for i, name in enumerate(["prob1", "prob2"]):
            events = set_ak_column(events, f"{self.cls_name}.{name}", outputs[2+i])


        print(events[self.cls_name])
        return events


# usable derivations
example = SpaNetModel.derive("spanet", cls_dict={"folds": 1})
