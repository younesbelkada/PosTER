import json

from PosTER.Agents.predictor_agent import Predictor_Agent
from PosTER.Agents.utils_agent import get_dataset
from PosTER.Datasets.titan_dataset import Sequence, Frame, Person

with open('config.json', 'r') as f:
    config = json.load(f)

predictor = Predictor_Agent(config, 'input')
predictor.predict()