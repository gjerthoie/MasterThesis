# Master's Thesis: Investigating Generalizability of Machine Learning Models for Human Activity Recognition Based on Wrist-Worn Accelerometer Data

##  Abstract 
Health tracking for the elderly population is an increasingly important challenge, with the potential to support remote care and improve health outcomes. However, the lack of publicly available datasets representing older adults makes the development of accurate and robust models for this group complicated. Most existing data is either private or collected from younger populations, thus limiting generalizability. This thesis looks into how classical machine learning models can generalize across users, age groups, genders, and datasets using only wrist worn accelerometer data. Public datasets such as GOTOV, PAMAP2, and CAPTURE-24 were evaluated to simulate a real world scenario. The results highlight a significant drop in accuracy when models are applied to unseen users or across datasets. However, the inclusion of a small portion (5\%) of user specific data during post training drastically improved performance. The findings demonstrate both the potential and limitations of classical models in elderly human activity recognition and emphasize the need for personalization and better representative datasets.

##  Research Question
How well does classical machine learning models generalize by utilizing wrist worn sensors, and how can limited personalization or post-training improve performance across users and datasets, with particular focus on elderly population?
##  Repository Structure

## Project Structure
MasterThesis/
│
├── CAPTURE 24 test/
│   ├── Age_GenderTEST.ipynb
│   ├── BASEMODELS_CAPTURE24.ipynb
│   └── Preprocess/
│       ├── preprosess-.ipynb
│       ├── SeeLabels.ipynb
│       └── windowpx.ipynb
│
├── generalized data/
│   └── TRAIN_GOTOV_PAMAP_TEST_CAPTURE.ipynb
│
├── GOTOV spesific/
│   ├── All_models_GOTOV.ipynb
│   ├── CrossvalidateGOTOV.ipynb
│   ├── Downsample.ipynb
│   ├── PosttrainGOTOV.ipynb
│   ├── Set_up_for_performing.py
│   ├── Upsample.ipynb
│   └── Preprocessing for GOTOV/
│       ├── Create_train_test_split.ipynb
│       ├── Create_windows.ipynb
│       └── preprosess.ipynb
│
├── PAMAP test/
│   ├── BASEMODELS_PAMAP.ipynb
│   └── Preprocess/
│       ├── preprosess.ipynb
│       └── window_creator.ipynb
│
├── Set_up_for_performing.py
├── README.md
└── .git/


 Datasets
This project uses the following publicly available datasets:

CAPTURE-24
Description: Wrist-worn accelerometer data collected in free-living conditions.
Link: [https://datashare.ed.ac.uk/handle/10283/3192](https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001)

PAMAP2 Physical Activity Monitoring
Description: Activities recorded from multiple body locations.
Link: https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring

GOTOV Dataset
Description: Multi-sensor dataset for older individuals.
Link: [https://osf.io/h4qwa](https://data.4tu.nl/articles/dataset/GOTOV_Human_Physical_Activity_and_Energy_Expenditure_Dataset_on_Older_Individuals/12716081)
