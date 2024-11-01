"""
This is a boilerplate pipeline 'feature_eng'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import features_eng, selected_features

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=features_eng,
            inputs=['intermediate_lc_clean', 'primary_analysis', 'parameters'],
            outputs='features_new',
            name='features_eng_node',
            tags='Features'
        ),
        node(
            func=selected_features,
            inputs=['primary_analysis', 'features_new', 'parameters'],
            outputs='features_dataset',
            name='selected_features_node',
            tags='Features'
        )
    ]) # type: ignore
