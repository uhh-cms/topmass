![](https://github.com/uhh-cms/topmass/actions/workflows/lint_and_test.yaml/badge.svg)

# topmass_alljets Analysis


### Resources

- [columnflow](https://github.com/columnflow/columnflow)
- [law](https://github.com/riga/law)
- [order](https://github.com/riga/order)
- [luigi](https://github.com/spotify/luigi)

## Example commands for cf.PlotVariables1D workflow:

### Local (not recommended for full datasets):
```
law run cf.PlotVariables1D --version [version name] --datasets 'data*',tt_fh_powheg,'qcd*' --variables jet6_pt
```

### HTCondor:
```
law run cf.PlotVariables1D --version [version name] --datasets 'data*',tt_fh_powheg,'qcd*' --variables jet6_pt 
--cf.MergeHistograms-{workflow=htcondor,htcondor-memory=3000} --cf.ReduceEvents-{workflow=htcondor,htcondor-memory=3000}
```

### Trigger efficiency plots (bin_sel variable dictates the trigger choice):
```
law run cf.PlotVariables1D --version [version name] --datasets data_jetht_d,data_jetht_e,data_jetht_f,tt_fh_powheg --variables jet6_pt-ht1-trig_bits 
--selector trigger_eff --producers default,trigger_prod --plot-function alljets.plotting.trigger_eff_plot_procs_binned.plot_efficiencies
--cf.MergeHistograms-{workflow=htcondor,htcondor-memory=3000} --cf.ReduceEvents-{workflow=htcondor,htcondor-memory=3000} 
--general-settings "bin_sel=1"
```
