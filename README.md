![](https://github.com/uhh-cms/topmass/actions/workflows/lint_and_test.yaml/badge.svg)

# Topmass Alljets Analysis
Analysis based on [columnflow](https://github.com/columnflow/columnflow), [law](https://github.com/riga/law) and [order](https://github.com/riga/order).

## Quickstart
A couple test tasks are listed below. They might require a valid voms proxy for accessing input data.
```
# clone the project 
git clone --recursive git@github.com:uhh-cms/topmass.git
cd alljets

# source the setup and store decisions in .setups/dev.sh (arbitrary name)
source setup.sh dev

# run your first task
law run cf.ReduceEvents \
    --version v1 \
    --dataset tt_fh_powheg \
    --branch 0

law run cf.PlotVariables1D \
    --version v1 \
    --datasets tt_fh_powheg \
    --producers features \
    --variables jet6_pt,n_jet \
    --categories 6j
```

## Most important files / building blocks

### Config
The most relavant files for the configuration of the analysis can be found [here](https://github.com/uhh-cms/topmass/tree/dev_Lennert/alljets/config). These files define the main analysis object, its configs, variables and categories. The main configuration happens [here](https://github.com/uhh-cms/topmass/blob/dev_Lennert/alljets/config/configs_aj.py) and defines the used datasets, processes, shifts, default calibrators/selectors/producers and much more. The default config is ```2017_v9```. For testing, it is recommended to use ```2017_v9_limited``` for reduced event statistics. The config can be passed to the task via the ```--config``` parameter, e.g.
```
law run cf.SelectEvents --version v1 --config 2017_v9
```

### Selectors
Modules that are used to define event selections are usually named ```selectors``` and can be found [here](https://github.com/uhh-cms/topmass/tree/dev_Lennert/alljets/selection). The two main selectors are called ```trigger_eff``` and ```default_trig_weight``` and are defined [here](https://github.com/uhh-cms/topmass/blob/dev_Lennert/alljets/selection/default.py). You can call the SelectEvents task using a selector, e.g. via
```
law run cf.SelectEvents --version v1 --selector default_trig_weight 
```

### Producers
Modules that are used to produce new columns are usually named ```producers``` and can be found in [here](https://github.com/uhh-cms/topmass/tree/dev_Lennert/alljets/production). A producer can be used as a part of another calibrator/selector/producer. 
We have several producers and these can be found [here](https://github.com/uhh-cms/topmass/blob/dev_Lennert/alljets/production/default.py). You can call a producer on its own as part of the ```ProduceColumns``` task, e.g. via
```
law run cf.ProduceColumns --version v1 --producers kinFitMatch
```
which performs a kinematic fit and produces new columns related to it.

### Law config
The law config is located [here](https://github.com/uhh-cms/topmass/blob/dev_Lennert/law.cfg) and takes care of information available to law when calling tasks. In this config, we can set some defaults for parameters and define, in which file system to store outputs for each task and which tasks should be loaded for the analysis.


## Workflows
In the following, the main workflows are briefly explained and example commands are provided. 

### Trigger correction
To correct for the difference of the trigger efficiencies between MC and data, a correction weight (```trigger_weight```) is determined via the ```ProduceTriggerWeight``` task in [here](https://github.com/uhh-cms/topmass/tree/dev_Lennert/alljets/tasks). This weight is currently determined from a 1D observable (```jet6_pt_trigger```) but others observables related to the trigger are also considerable like ```ht_trigger```. 

If you choose to start with the selector ```default_trig_weight```, this task is automatically triggered and produces the correction weights with the settings defined in [here](https://github.com/uhh-cms/topmass/blob/dev_Lennert/alljets/production/trig_cor_weight.py). 

But first, the trigger efficiency curves, which can be obtained via the command, e.g. for the variable ```jet6_pt_trigger```
```
law run cf.PlotVariables1D --version v1 --configs 2017_v9 \ 
    --datasets tt_fh_powheg,tt_dl_powheg,tt_sl_powheg,'data*' \
    --selector trigger_eff \
    --selector-steps All,BaseTrigger,BTag,HT \   
    --producers no_norm,trigger_prod \ 
    --variables jet6_pt_trigger-trig_bits \
    --hist-producer trig_all_weights \ 
    --processes data,tt \
    --categories incl \
    --plot-function alljets.plotting.trigger_eff_closure_1D.plot_efficiencies \
    --general-settings "bin_sel=1"
```
Note that for the variable ```ht_trigger```, we would change the selector-steps to ```All,BaseTrigger,SixJets,BTag,jet```.

Now, to produce the correction weight, we can call the ```ProduceTriggerWeight``` task via
```
law run cf.ProduceTriggerWeight --version v1 --configs 2017_v9 \
    --datasets tt_fh_powheg,tt_dl_powheg,tt_sl_powheg,'data*' \ 
    --selector trigger_eff \ 
    --selector-steps All,BaseTrigger,BTag,HT \
    --producers no_norm,trigger_prod \
    --variables jet6_pt_trigger-trig_bits \
    --hist-producer trig_all_weights \
    --categories incl \
    --general-settings "bin_sel=1,unweighted=0
```

To check the effect of this trigger correction weight, we can plot the variables once again. For this, we have to change the selector to ```default_trig_weight``` as there the weight is passed, but we keep the ```--selection-steps``` for each observable respectively. The command to obtain the plots with the applied correction and an estimated uncertainty is, e.g.

```
law run cf.PlotShiftedVariables1D --version v1 --configs 2017_v9 \
    --datasets tt_fh_powheg,tt_dl_powheg,tt_sl_powheg,'data*' \
    --selector default_trig_weight \
    --selector-steps All,BaseTrigger,BTag,HT \
    --producers no_norm,trigger_prod \
    --variables jet6_pt_trigger-trig_bits \ 
    --hist-producer trig_all_weights \
    --processes data,tt \
    --categories incl \
    --plot-function alljets.plotting.trigger_eff_closure_1D.plot_efficiencies_with_uncert \
    --general-settings "bin_sel=1" \
    --shift-sources trig 
``` 
It should be noted that for these plots the ```--hist-producer trig_all_weight``` is needed and can be ignored in the following.

### Kinematic Fit
After the production of the trigger correction, the event selection using the ```default_trig_weight``` selector and a kinematic fit using the ```kinFitMatch``` producer is performed. After the fit, we can plot variables like ```reco_Top1_mass``` or ```fit_Top1_mass``` or any other found [here](https://github.com/uhh-cms/topmass/blob/dev_Lennert/alljets/config/variables.py). 

However, we are more interested in the matching performance of the fit. To plot the variables with the matching, we use a custom plotting function found [here](https://github.com/uhh-cms/topmass/blob/dev_Lennert/alljets/plotting/plot_hist_matching.py) and call it, e.g via

```
law run cf.PlotVariables1D --version v1 --configs 2017_v9 \
    --datasets tt_fh_powheg,tt_dl_powheg,tt_sl_powheg \
    --selector default_trig_weight \
    --producers default,kinFitMatch \
    --variables fit_Top1_mass-fit_combination_type \
    --categories incl \
    --plot-function alljets.plotting.plot_hist_matching.plot_hist_matching_MC
```

### Data Driven Background Estimation
The phase space region of the fully hadronic decay channel is dominated by the QCD multijet background. However, the MC simulation for this background suffers from large cross sections and large event weights. This leads to jagged distribution when using the MC samples for the QCD background. Thus, a data-driven background estimation is used.

However, we can first plot the distributions of the variables including the QCD simulation and data. For this, we use another custom plotting function but the command is very similar structured as above, 
```
law run cf.PlotVariables1D --version v1 --configs 2017_v9 \
    --datasets tt_fh_powheg,tt_sl_powheg,tt_dl_powheg,'data*','qcd*' \
    --selector default_trig_weight \
    --producers default,kinFitMatch
    --variables fit_Top1_mass-fit_combination_type \
    --processes tt,qcd,data \
    --categories incl \
    --plot-function alljets.plotting.plot_hist_matching.plot_hist_matching
```

To derive a background estimation, we need to isolate that background from the signal. For this, we use a different group of ```selector-steps``` and need to invoke the ```--hist-hook qcd``` parameter, which is defined [here](https://github.com/uhh-cms/topmass/blob/dev_Lennert/alljets/hist_hooks/bkg.py). Then, we can obtain the same plot above using the data-driven QCD background estimation instead the MC, via

```
law run cf.PlotVariables1D --version v1 --configs 2017_v9 \
    --datasets tt_fh_powheg,tt_sl_powheg,tt_dl_powheg,'data*' \
    --selector default_trig_weight \
    --selector-steps All,SignalOrBkgTrigger,BTag20,jet,HT \
    --producers default,kinFitMatch \
    --variables fit_Top1_mass-fit_combination_type \
    --processes tt,qcd_est,data \
    --hist-hook qcd \
    --categories sig \
    --plot-function alljets.plotting.plot_hist_matching.plot_hist_matching 
```

## Debugging

The output path of the different task can be yielded by appending ```print-output <index>```. For the output of the main task the index is 0, e.g. where the plots of the ```cf.Plotvariables1D``` tasks are stored. We can increase the index number to follow the workflow tree top to bottom.

The output of the tasks can be inspected using the ```cf_inspect``` command provided from columnflow followed by the path to the file. This is available after sourcing the environment. 

For histograms, columnflow provides a dedicated task for the purpose of debugging. The task can be called via ```law run cf.InspectHistograms```.

For more detailed information see the [Columnflow documentation](https://columnflow.readthedocs.io/en/latest/index.html).

