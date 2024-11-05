"""
This is a boilerplate pipeline 'split_dataset'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import split_dataset, model_pipeline, train_model, evaluate_metrics


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_dataset,
            inputs= ['intermediate_lc_dataset', 'parameters'],
            outputs=['X_train#mi', 'X_test#mi', 'y_train#mi', 'y_test#mi'],
            name='split_dataset_node',
        ),
        node(
            func=model_pipeline,
            inputs='parameters',
            outputs='model_pipe',
            name='model_pipeline_node'
        ),
        node(
            func=train_model,
            inputs=['X_train#mi', 'y_train#mi', 'model_pipe', 'params:model_options'],
            outputs='regressor',
            name='train_model_node'
        ),
        node(
            func=evaluate_metrics,
            inputs=['regressor', 'params:model_options', 'X_test#mi', 'y_test#mi'],
            outputs=None,
            name='evaluate_metrics_node'
        ),
    ],
    tags='Model') # type: ignore
