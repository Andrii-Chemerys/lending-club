"""
This is a boilerplate pipeline 'split_dataset'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import split_balance

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_balance,
            inputs= ['features_dataset', 'parameters'],
            outputs=['X_train#mi', 'X_test#mi', 'y_train#mi', 'y_test#mi'],
            name='split_features_node',
        )
    ], tags="Inputs")
