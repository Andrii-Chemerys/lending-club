"""
This is a boilerplate pipeline 'data_clean'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import clean_dataset

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=clean_dataset,
            inputs=['intermediate_lc_dataset','parameters'],
            outputs="primary_lc_dataset",
        )
    ])
