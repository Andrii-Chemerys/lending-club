"""
This is a boilerplate pipeline 'feature_eng'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import  selected_features

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=selected_features,
            inputs=['primary_analysis', 'parameters'],
            outputs='model_features#yml',
            name='selected_features_node',
            tags='Features'
        )
    ]) # type: ignore
