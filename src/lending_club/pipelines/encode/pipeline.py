"""
This is a boilerplate pipeline 'encode'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import encode_dataset

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=encode_dataset,
            inputs=['primary_lc_dataset_outl_log', 'parameters'],
            outputs='primary_lc_dataset_encoded',
        )
    ])
