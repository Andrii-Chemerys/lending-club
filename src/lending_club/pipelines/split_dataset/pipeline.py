"""
This is a boilerplate pipeline 'split_dataset'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from sklearn.model_selection import train_test_split
from .nodes import split_features_target


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_features_target,
            inputs= 'features_dataset',
            outputs= ['features_X', 'features_y'],
            name='split_features_target_node',
            tags='Features'
        ),
        node(
            func=train_test_split,
            inputs=['features_X', 'features_y', 'params:test_size', 'params:random_state'],
            outputs=['features_X_train', 'features_X_test', 'features_y_train', 'features_y_test'],
            name='train_test_split_node',
            tags='Features'
        )
    ]) # type: ignore
