"""
This is a boilerplate pipeline 'feature_eng'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import features_eng

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=features_eng,
            inputs=['intermediate_lc_clean', 'primary_analysis', 'parameters'],
            outputs='features_dataset',
            name='features_eng_node',
            tags='Features'
        )
    ]) # type: ignore
