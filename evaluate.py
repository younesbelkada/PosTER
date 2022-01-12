import json

from PosTER.Agents.evaluator_agent import Evaluator
from PosTER.Agents.utils_agent import get_test_dataset
from PosTER.Datasets.titan_dataset import Sequence, Frame, Person

with open('config.json', 'r') as f:
    config = json.load(f)

test_dataloader = get_test_dataset(config)

evaluator = Evaluator(config)
evaluator.evaluate(test_dataloader)