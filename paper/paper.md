---
title: "CosinorAge: Unified Python and Web Platform for Biological Age Estimation from Wearable- and Smartwatch-Based Activity Rhythms"
tags:
  - Python
  - circadian rhythms
  - biological age
  - digital health
  - wearables
authors:
  - name: Jinjoo Shim
    orcid: 0000-0000-0000-0000
    corresponding: true
    affiliation: "1, 2"
  - name: Jacob Hunecke
    affiliation: 2
  - name: Elgar Fleisch
    affiliation: "2, 3"
  - name: Filipe Barata
    affiliation: 2
affiliations:
 - name: Department of Biostatistics, Harvard University, Cambridge, MA, USA
   index: 1
 - name: Centre for Digital Health Interventions, ETH Zurich, Zurich, Switzerland
   index: 2
 - name: Centre for Digital Health Interventions, University of St.Gallen, St.Gallen, Switzerland
   index: 3
date: 2025-08-27
bibliography: paper.bib
---

# Summary

Every day, millions of people worldwide track their steps, sleep, and activity rhythms with smartwatches and fitness trackers. These continuously collected data streams present a remarkable opportunity to transform routine self-tracking into meaningful health insights that enable individuals to understand—and potentially influence—their biological aging. Yet most tools for analyzing wearable data remain fragmented, proprietary, and inaccessible, creating a major barrier between this vast reservoir of personal health information and its translation into actionable insights on aging.

`CosinorAge` is an open-source framework that estimates biological age from wearable-derived circadian, physical activity, and sleep metrics. It addresses the lack of unified, reproducible pipelines for jointly analyzing rest–activity rhythmicity, physical activity, and sleep, and linking them to health outcomes. The Python package provides an end-to-end workflow from raw data ingestion and preprocessing to feature computation and biological age estimation, supporting multiple input sources across wearables and smartwatch. Its companion web-based `CosinorAge` Calculator enables non-technical users to access identical analytical capabilities through an intuitive interface. By combining transparent, reproducible analysis with broad accessibility, `CosinorAge` advances scalable, personalized health monitoring and bridges digital health technologies with biological aging research.

# Statement of Need

Circadian rhythms play a critical role in maintaining key regulatory systems, including metabolic, immune, and endocrine pathways, and tightly govern rest–activity cycles encompassing sleep and physical activity, both essential to healthy aging. Disruptions in these daily rhythms, such as reduced amplitude, irregular activity timing, low activity levels, or poor sleep regularity, have been consistently linked to increased risk of chronic diseases, mortality, systemic inflammation, and accelerated biological aging [@shim2024circadian; @shim2025wrist]. Given these established links, there is an urgent need for continuous high-resolution monitoring of daily activity-rest patterns to characterize individualized rhythmicity profiles, detect early deviations that signal elevated risk, and guide timely targeted interventions to optimize health span and slow biological aging.  

Wearable devices and smartwatches enable a scalable, non-invasive, and cost-efficient method for digital biomarkers of circadian rhythms, physical activity, and sleep at both individual and population levels. However, existing analytic tools to analyze wearable data typically focus on isolated metric extraction or rely on proprietary algorithms, which limits transparency, reproducibility, and their ability to inform downstream health outcomes, such as biological age. To address this gap, we developed `CosinorAge` [@shim2024circadian], a digital biomarker framework that estimates biological age and healthspan from circadian rest-activity rhythms using wearables.  

Currently, there is no open-source analytic pipeline that offers a unified, end-to-end workflow for processing raw wearable data from data reading and pre-processing to feature computation and biological age estimation. Although circadian rhythm, physical activity, and sleep are physiologically interdependent and should be analyzed jointly, existing packages such as pyActigraphy [@hammad2021pyactigraphy], actipy [@actipy], CosinorPy [@movskon2020cosinorpy], and scikit-digital-health [@adamowicz2022scikit] are limited to analyzing specific domains. While the GGIR R package [@ggir] provides extensive analysis of accelerometer data, it is only available in R and does not provide functionality to further link metrics to health outcomes. Moreover, these tools are designed primarily for technical audiences, with no accessible interface for non-experts to upload and analyze their own data. This lack of accessibility limits the translational potential of scientific insights and leaves individuals dependent on opaque, proprietary manufacturer apps with limited interpretability.  

In response, we developed the **CosinorAge Python Package** and **CosinorAge Calculator**. The Python Package provides a fully integrated workflow that processes raw accelerometer data, extracts features related to circadian rhythms, physical activity, and sleep, and estimates biological age, `CosinorAge`, as shown in \autoref{fig:Fig1}. Importantly, `CosinorAge` represents a completely digital *second-order clock*, derived from mortality risk rather than chronological age, thereby capturing healthspan-relevant biological processes with greater precision [@shim2024circadian]. The workflow supports input from a range of research-grade and consumer wearable devices, including research-grade actigraphy, large-scale population resources such as UK Biobank and NHANES, as well as consumer-centric smartwatches like the Samsung Galaxy Watch. Recent work has validated comparability between research-grade actigraphy and consumer smartwatches in assessing circadian rhythms, underscoring the potential of `CosinorAge` to integrate data across device types and study contexts [@wu2025comp].  

What makes our contribution particularly valuable is its scalability and accessibility: wearables are already widespread, cost-efficient, and continuously collecting high-resolution behavioral data. By leveraging this infrastructure, **CosinorAge Calculator** allows anyone—even without technical expertise—to estimate their biological age from their own wearable data. This democratization of health analytics opens a path to personalized monitoring of aging and resilience, bridging the gap between population-level research and personal health insights.  

The **CosinorAge Python Package** and **CosinorAge Calculator** are applicable to studies where wearable-derived behavioral rhythms are central to the research question. They can be used to quantify circadian, activity, and sleep patterns in terms of strength, timing, and stability, for example in studies examining whether rhythm disruptions are associated with aging trajectories or adverse health outcomes. When the research question concerns biological age, the package provides estimates from wearable data that can be incorporated into analyses of rhythm characteristics, aging processes, or intervention effects.  

![Minute-level activity data collected using a Samsung Galaxy smartwatch from a 45-year-old female over 7 days was analyzed using `CosinorAge` Python package. The blue lines display ENMO activity intensity, while the red curve indicates the cosinor model fit. Green and red shading mark wear and non-wear periods. Based on the recorded activity pattern, the predicted biological age is 49.0 years.\label{fig:Fig1}](figures/timerseries&CA.png)

# How it Works

## CosinorAge Python Package

The **CosinorAge Python package** is structured into three core modules, each representing a key stage in the pipeline for analyzing accelerometer data and predicting biological age, `CosinorAge`. Its modular architecture allows components to be used independently or integrated into a streamlined workflow.  

![Package scheme.\label{fig:Fig2}](figures/schema.png)

### DataHandler Module
Supports accelerometer data from UK Biobank (UKB), NHANES, Samsung Galaxy Smartwatches, and Bring-Your-Own-Data (BYOD). Each performs filtering, preprocessing, and scaling to produce standardized minute-level ENMO time series. A GenericDataHandler allows CSV-based BYOD analysis.  

### WearableFeatures Module
Includes two classes: `WearableFeatures` (individual-level) and `BulkWearableFeatures` (batch cohort-level). Computes physical activity, sleep, and circadian rhythm metrics such as MESOR, amplitude, acrophase, M10, L5, IS, IV, RA, TST, WASO, PTA, NWB, and SOL.  

| **Domain** | **Metrics** |
|------------|-------------|
| Circadian Rhythm Analysis | MESOR, amplitude, acrophase, M10, L5, IS, IV, RA |
| Physical Activity Analysis | LPA, MPA, VPA, sedentary duration |
| Sleep Analysis | TST, WASO, PTA, NWB, SOL |

### CosinorAge Module
Final stage of the pipeline: predicts the `CosinorAge` biomarker using a proportional hazards model. Supports unisex, female-specific, and male-specific coefficients.  

## CosinorAge Calculator: Web User Interface

The web-based interface (www.cosinorage.app) allows non-technical users to analyze their data with the same backend package. It provides an overview, documentation, and an interactive calculator with CSV uploads, cohort summaries, correlation matrices, and visual dashboards.  

![Data upload interface & Summary dashboard.](figures/calc.png)

# References