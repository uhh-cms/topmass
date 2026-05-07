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
Modules that are used to define event selections are usually named ```selectors``` and can be found [here](https://github.com/uhh-cms/topmass/tree/dev_Lennert/alljets/selection). The two main selectors are called [```trigger```](https://github.com/uhh-cms/topmass/blob/dev_Lennert/alljets/selection/trigger.py) and [```default```](https://github.com/uhh-cms/topmass/blob/dev_Lennert/alljets/selection/default.py). You can call the SelectEvents task using a selector, e.g. via
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
Law configs files tell the workflow what to load, where data lives, and which versions to use. [law.cfg](https://github.com/uhh-cms/topmass/blob/dev_Lennert/law.cfg) controls workflow defaults and behavior; [law_fs.cfg](https://github.com/uhh-cms/topmass/blob/dev_Lennert/law_fs.cfg) defines storage endpoints and mounts; [law_outputs.cfg](https://github.com/uhh-cms/topmass/blob/dev_Lennert/law_outputs.cfg) routes task outputs and pins versions for reproducibility.


## Workflows
In the following, the main workflows are briefly explained and example commands are provided. 

### Trigger correction
To correct for the difference of the trigger efficiencies between MC and data, a correction weight (```trigger_weight```) is determined via the ```ProduceTriggerWeight``` task in [here](https://github.com/uhh-cms/topmass/tree/dev_Lennert/alljets/tasks). This weight is currently determined from a 1D observable (```trigJet6_pt```) but others observables related to the trigger are also considerable like ```ht_trigger```. 

If you choose to start with the selector ```default```, this task is automatically triggered and produces the correction weights with the settings defined in [here](https://github.com/uhh-cms/topmass/blob/dev_Lennert/alljets/production/trig_cor_weight.py). 

But first, the trigger efficiency curves, which can be obtained via the command, e.g. for the variable ```trigjet6_pt```
```
law run cf.PlotVariables1D --version v1 --configs 2017_v9 \ 
    --datasets tt_fh_powheg,tt_dl_powheg,tt_sl_powheg,'data*' \
    --selector trigger\
    --selector-steps BaseTrigger,BTag,HT \   
    --producers trigSF_prod,trigger_prod \ 
    --variables trigjet6_pt-trig_bits \
    --hist-producer trig_all_weights \ 
    --processes data,tt \
    --categories incl \
    --plot-function alljets.plotting.trigger_eff_closure_1D.plot_efficiencies \
    --general-settings "bin_sel=1"
```
Note that for the variable ```ht_trigger```, we would change the selector-steps to ```BaseTrigger,SixJets,BTag,jet```.

Now, to produce the correction weight, we can call the ```ProduceTriggerWeight``` task via
```
law run cf.ProduceTriggerWeight --version v1 --configs 2017_v9 \
    --datasets tt_fh_powheg,tt_dl_powheg,tt_sl_powheg,'data*' \ 
    --selector trigger \ 
    --selector-steps BaseTrigger,BTag,HT \
    --producers trigSF_prod,trigger_prod \
    --variables trigjet6_pt-trig_bits \
    --hist-producer trig_all_weights \
    --categories incl \
    --general-settings "bin_sel=1,unweighted=0
```

To check the effect of this trigger correction weight, we can plot the variables once again. For this, we have to change the producer from ```trigSF_prod``` to ```trigSF_eval``` as there the trigger correction weight is passed. The command to obtain the plots with the applied correction and an estimated uncertainty is, e.g.

```
law run cf.PlotShiftedVariables1D --version v1 --configs 2017_v9 \
    --datasets tt_fh_powheg,tt_dl_powheg,tt_sl_powheg,'data*' \
    --selector trigger \
    --selector-steps BaseTrigger,BTag,HT \
    --producers trigSF_eval,trigger_prod \
    --variables trigjet6_pt-trig_bits \ 
    --hist-producer trig_all_weights \
    --processes data,tt \
    --categories incl \
    --plot-function alljets.plotting.trigger_eff_closure_1D.plot_efficiencies_with_uncert \
    --general-settings "bin_sel=1" \
    --shift-sources trig 
``` 
It should be noted that for these plots the ```--hist-producer trig_all_weight``` is needed and can be ignored in the following.

### Kinematic Fit
After the production of the trigger correction, the event selection using the ```default``` selector and a kinematic fit using the ```kinFitMatch``` producer is performed. After the fit, we can plot variables like ```reco_Top1_mass``` or ```fit_Top1_mass``` or any other found [here](https://github.com/uhh-cms/topmass/blob/dev_Lennert/alljets/config/variables.py). 

However, we are more interested in the matching performance of the fit. To plot the variables with the matching, we use a custom plotting function found [here](https://github.com/uhh-cms/topmass/blob/dev_Lennert/alljets/plotting/plot_hist_matching.py) and call it, e.g via

```
law run cf.PlotVariables1D --version v1 --configs 2017_v9 \
    --datasets tt_fh_powheg,tt_dl_powheg,tt_sl_powheg \
    --selector default \
    --selector-steps Trigger,Lepton_Veto,HT,jet,BTag,LeadingSix2BTag \
    --producers default,kinFitMatch \
    --variables fit_Top1_mass-fit_combination_type \
    --categories sig \
    --plot-function alljets.plotting.plot_hist_matching.plot_hist_matching_MC
```

### Data Driven Background Estimation
The phase space region of the fully hadronic decay channel is dominated by the QCD multijet background. However, the MC simulation for this background suffers from large cross sections and large event weights. This leads to jagged distribution when using the MC samples for the QCD background. Thus, a data-driven background estimation is used.

However, we can first plot the distributions of the variables including the QCD simulation and data. For this, we use another custom plotting function but the command is very similar structured as above, 
```
law run cf.PlotVariables1D --version v1 --configs 2017_v9 \
    --datasets tt_fh_powheg,tt_sl_powheg,tt_dl_powheg,'data*','qcd*' \
    --selector default \
    --producers default,kinFitMatch
    --variables fit_Top1_mass-fit_combination_type \
    --processes tt,qcd,data \
    --categories sig \
    --plot-function alljets.plotting.plot_hist_matching.plot_hist_matching
```
Before applying the background estimation, we need to verify its performance using the QCD multijet MC simulation. This ensures that the background selection and the signal selection, result in distributions of the same shape for multijet events, which do originate from $t\bar{t}$ events. These validation plots, for example for the ``` variable fit_Top1_mass``` via

```
law run cf.PlotVariables1D --version v1 --configs 2017_v9 \
    --datasets '*qcd*' \
    --selector default \
    --producers default,kinFitMatch,trigger_prod \
    --variables fit_Top1_mass-secmaxbtag_type-trig_bits \
    --processes qcd \
    --categories fit_conv_leq_rbb \
    --plot-function alljets.plotting.sim_vs_est.qcd_sig_vs_bkg_sel
```
After this, we can then compare the distributions of our variables for the QCD MC and the data-driven background estimation in our signal region via the command

```
law run cf.PlotVariables1D --version v1 --configs 2017_v9 \
    --datasets tt_fh_powheg,tt_sl_powheg,tt_dl_powheg,'data*','qcd*' \
    --variables fit_Top1_mass \
    --selector default\
    --processes tt,data,qcd,qcd_est
    --categories sig \
    --plot-function alljets.plotting.sim_vs_est.qcd_mc_vs_est \
    --hist-hook qcd
```

To derive a background estimation, we need to isolate that background from the signal. For this, we need to invoke the ```--hist-hook qcd``` parameter, which is defined [here](https://github.com/uhh-cms/topmass/blob/dev_Lennert/alljets/hist_hooks/bkg.py). Then, we can obtain the same plot above using the data-driven QCD background estimation instead the MC, via

```
law run cf.PlotVariables1D --version v1 --configs 2017_v9 \
    --datasets tt_fh_powheg,tt_sl_powheg,tt_dl_powheg,'data*' \
    --selector default \
    --producers default,kinFitMatch \
    --variables fit_Top1_mass-fit_combination_type \
    --processes tt,qcd_est,data \
    --hist-hook qcd \
    --categories sig \
    --plot-function alljets.plotting.plot_hist_matching.plot_hist_matching 
```

### Inference

To create the 1D data cards:
```
law run cf.CreateDatacards --inference-model default_1D --hist-hooks qcd   --version v1_Analysis  --configs 2017_v9  --selector-steps All,SignalOrBkgTrigger,HT,jet,BTag20,LeadingSix20BTag --workers 8 --tasks-per-job 10 
```

To create the 2D data card:
```
 law run cf.CreateDatacards --inference-model default_2D --hist-hooks qcd,unrolling_2D   --version v1_Analysis  --configs 2017_v9  --selector-steps All,SignalOrBkgTrigger,HT,jet,BTag20,LeadingSix20BTag --workers 8 --tasks-per-job 10 --remove-output 0,a
added config:2017_v9
```

## Debugging

The output path of the different task can be yielded by appending ```print-output <index>```. For the output of the main task the index is 0, e.g. where the plots of the ```cf.Plotvariables1D``` tasks are stored. We can increase the index number to follow the workflow tree top to bottom.

The output of the tasks can be inspected using the ```cf_inspect``` command provided from columnflow followed by the path to the file. This is available after sourcing the environment. 

For histograms, columnflow provides a dedicated task for the purpose of debugging. The task can be called via ```law run cf.InspectHistograms```.

For more detailed information see the [Columnflow documentation](https://columnflow.readthedocs.io/en/latest/index.html).

