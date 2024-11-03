"""
This is a boilerplate pipeline 'split_dataset'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import model_pipeline, train_model, evaluate_metrics


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=model_pipeline,
            inputs='parameters',
            outputs='model_pipe',
            name='model_pipeline_node'
        ),
        node(
            func=train_model,
            inputs=['X_train', 'y_train', 'model_pipe'],
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
