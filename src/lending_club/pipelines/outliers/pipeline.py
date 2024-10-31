"""
This is a boilerplate pipeline 'outliers'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import _outliers_handler

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=outliers_handler,
            inputs= ['primary_lc_dataset', 'parameters'],
            outputs= 'primary_lc_dataset_outl_log',
        )
    ])
