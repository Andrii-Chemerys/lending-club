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
            inputs=['intermediate_lc_clean', 'parameters'],
            outputs='intermediate_encoded',
            name='encode_dataset_node',
            tags='Intermediate'
        )
    ]) # type: ignore
