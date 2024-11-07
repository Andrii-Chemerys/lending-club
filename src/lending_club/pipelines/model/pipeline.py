"""
This is a boilerplate pipeline 'split_dataset'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import split_dataset, model_pipeline, train_model, evaluate_metrics, _drop_duplicates

from lending_club.pipelines.data_processing.nodes import processing_lc
from lending_club.pipelines.analysis.nodes import features_eng

# Full pipeline for model fitting
'''
def create_pipeline(**kwargs) -> Pipeline:
    pipe_instance = pipeline([
        node(
            func=processing_lc,
            inputs='original_lc_dataset',
            outputs='intermediate_lc_dataset',
            name='model_processing_lc_node',
            ),
        node(
            func=_drop_duplicates,
            inputs='intermediate_lc_dataset',
            outputs='intermediate_lc_unique',
            name='model_drop_duplicates_node',
            ),
        node(
            func=features_eng,
            inputs=['intermediate_lc_unique', 'parameters'],
            outputs='lc_new_features',
            name='model_features_eng_node',
            ),
        node(
            func=split_dataset,
            inputs=['intermediate_lc_unique', 'lc_new_features', 'parameters'],
            outputs=['X_train', 'X_test', 'y_train', 'y_test'],
            name='model_split_dataset_node',
        ),
        node(
            func=model_pipeline,
            inputs=['params:model_options' ,'parameters'],
            outputs='model_pipe',
            name='model_pipeline_node'
        ),
        node(
            func=train_model,
            inputs=['X_train', 'y_train', 'model_pipe', 'params:model_options'],
            outputs='regressor',
            name='train_model_node'
        ),
        node(
            func=evaluate_metrics,
            inputs=['regressor', 'params:model_options', 'X_test', 'y_test'],
            outputs=None,
            name='evaluate_metrics_node'
        ),
    ],
    tags='Model'
    ) # type: ignore

    pipeline_rfc = pipeline(
        pipe=pipe_instance,
        inputs='original_lc_dataset',
        namespace='baseline_model'
    ) # type: ignore

    pipeline_ctb = pipeline(
        pipe=pipe_instance,
        inputs='original_lc_dataset',
        namespace='candidate_model'
    ) # type: ignore

    return pipeline_rfc + pipeline_ctb

'''
