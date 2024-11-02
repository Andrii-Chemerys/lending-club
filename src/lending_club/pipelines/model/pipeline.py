"""
This is a boilerplate pipeline 'split_dataset'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import standartizer, train_model, evaluate_metrics


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_model,
            inputs=['X_train', 'y_train', 'params:model_options'],
            outputs='regressor',
            name='train_model_node'
        ),
        node(
            func=evaluate_metrics,
            inputs=['regressor', 'params:model_name', 'X_test', 'y_test'],
            outputs=None,
            name='evaluate_metrics_node'
        ),        
    ], 
    tags='Model') # type: ignore
