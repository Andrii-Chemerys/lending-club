"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node

from nodes import _parse_bool, _parse_date, _parse_pct, _parse_term

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(

        )
    ])
