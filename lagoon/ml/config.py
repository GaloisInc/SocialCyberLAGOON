import os
import torch


lagoon_artifacts_folder = os.path.join(
    os.path.dirname( #Parent directory containing the lagoon repo
        os.path.dirname( #lagoon/
            os.path.dirname( #lagoon/
                os.path.dirname(os.path.abspath(__file__)) #ml/
            )
        )
    ),
    'lagoon-artifacts'
)
assert os.path.isdir(lagoon_artifacts_folder), 'ERROR: lagoon-artifacts repository not found. Please clone the lagoon-artifacts repo from "https://gitlab-ext.galois.com/lagoon/lagoon-artifacts" as a sibling to the lagoon repo.'
DATA_FOLDER = os.path.join(lagoon_artifacts_folder, 'data')
RESULTS_FOLDER = os.path.join(lagoon_artifacts_folder, 'results')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSSFUNC_MAPPING = {
    'L1': torch.nn.L1Loss(),
    'L2': torch.nn.MSELoss()
}

TOXICITY_CATEGORIES = [f'computed_badwords_{category}' for category in ['googleInstantB_any', 'swearing_any', 'mrezvan94Harassment_Appearance', 'mrezvan94Harassment_Generic', 'mrezvan94Harassment_Intelligence', 'mrezvan94Harassment_Politics', 'mrezvan94Harassment_Racial', 'mrezvan94Harassment_Sexual']]

PEP_STATUSES = {
    'good': ['active', 'accepted', 'final', 'superseded'],
    'bad': ['withdrawn', 'rejected', 'deferred', 'april fool!'], # 'april fool!' was actually a rejected PEP
    'other': ['draft', 'provisional'] # 'draft' and 'provisional' PEPs can go both ways, hence are classified as 'other'
}

NLP_MODELS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nlp_models')