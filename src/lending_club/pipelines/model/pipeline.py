"""
This is a boilerplate pipeline 'split_dataset'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import split_features, standartizer, balancer, train_model, evaluate_metrics


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_features,
            inputs= ['features_dataset', 'parameters'],
            outputs=['__X_train', '_X_test', '_y_train', 'y_test'],
            name='split_features_node',
        ),
        node(
            func=standartizer,
            inputs=['__X_train','_X_test'],
            outputs=['_X_train','X_test'],
            name='standartizer_node',
        ),
        node(
            func=balancer,
            inputs=['_X_train', '_y_train'],
            outputs=['X_train', 'y_train'],
            name='balancer_node',
        ),
        node(
            func=train_model,
            inputs=['X_train', 'y_train', 'params:model_options'],
            outputs='regressor',
            name='train_model_node'
        ),
        node(
            func=evaluate_metrics,
            inputs=['regressor', 'params:model_name', 'X_test', 'y_test'],

            outputs='metrics',
            name='evaluate_metrics_node'
        ),        
    ], 
    tags='Model') # type: ignore
