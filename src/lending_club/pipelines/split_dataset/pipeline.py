"""
This is a boilerplate pipeline 'split_dataset'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import _split_features_target


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=_split_features_target,
            inputs= 'features_lc_dataset',
            outputs= ['X', 'y'],
            name='split_features_target_node',
            tags='Features'
        )
    ]) # type: ignore
