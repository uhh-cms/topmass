import awkward as ak
import numpy as np

mtfit = ak.from_parquet("""/data/dust/user/schaller/aj_store/analysis_aj/analysis_aj/cf.ProduceColumns/2017_v9/
tt_*_powheg/nominal/calib__default/sel__example_trig_weight/red__cf_default/prod__example/v1/columns_*.parquet""",
                        columns=["events", "FitTop1mass"])
reco_R_bq = ak.from_parquet("""/data/dust/user/schaller/aj_store/analysis_aj/analysis_aj/cf.ProduceColumns/2017_v9/
    tt_*_powheg/nominal/calib__default/sel__example_trig_weight/red__cf_default/prod__example/v1/columns_*.parquet""",
                            columns=["events", "Reco_R_bq"])
avg_W_mass = ak.from_parquet("""/data/dust/user/schaller/aj_store/analysis_aj/analysis_aj/cf.ProduceColumns/2017_v9/
    tt_*_powheg/nominal/calib__default/sel__example_trig_weight/red__cf_default/prod__example/v1/columns_*.parquet""",
                             columns=["events", "RecoW_avg_mass"])
categories = ak.from_parquet("""/data/dust/user/schaller/aj_store/analysis_aj/analysis_aj/cf.ProduceColumns/2017_v9/
    tt_*_powheg/nominal/calib__default/sel__example_trig_weight/red__cf_default/prod__example/v1/columns_*.parquet""",
                             columns=["events", "category_ids"])
mask = categories.category_ids == 1083385291
goodmask = ak.any(mask, axis=1)
# import IPython
# IPython.embed()
mtfit_flat = mtfit[goodmask].FitTop1mass
reco_R_bq_flat = reco_R_bq[goodmask].Reco_R_bq
avg_W_mass_flat = avg_W_mass[goodmask].RecoW_avg_mass
edges_mtfit = np.percentile(mtfit_flat, np.linspace(0, 100, 9))
edges_reco_R_bq = np.percentile(reco_R_bq_flat, np.linspace(0, 100, 9))
edges_avg_W_mass = np.percentile(avg_W_mass_flat, np.linspace(0, 100, 9))
print("mtfit : ", edges_mtfit)
print("Reco_R_bq : ", edges_reco_R_bq)
print("avg_W_mass : ", edges_avg_W_mass)
