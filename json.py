"""
Copyright 2023- The Outlines developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import mlx_lm

import json as pyjson
from typing import Callable, Optional, Union
from mlx_lm.models.llama import Model
from pydantic import BaseModel

from outlines.fsm.json_schema import (
    BOOLEAN,
    DATE,
    DATE_TIME,
    INTEGER,
    NULL,
    NUMBER,
    STRING,
    STRING_INNER,
    TIME,
    UUID,
    WHITESPACE,
    build_regex_from_schema,
    get_schema_from_signature,
    to_regex,
)

from functools import singledispatch
from outlinesmlx.samplers import Sampler, multinomial
from .regex import regex


@singledispatch
def json(
    model,
    schema_object: Union[str, object, Callable],
    max_tokens: Optional[int] = None,
    sampler: Sampler = multinomial(),
):
    if isinstance(schema_object, type(BaseModel)):
        schema = pyjson.dumps(schema_object.model_json_schema())
        regex_str = build_regex_from_schema(schema)
        generator = regex(model, regex_str, max_tokens, sampler)
        generator.format_sequence = lambda x: schema_object.parse_raw(x)
    elif callable(schema_object):
        schema = pyjson.dumps(get_schema_from_signature(schema_object))
        regex_str = build_regex_from_schema(schema)
        generator = regex(model, regex_str, max_tokens, sampler)
        generator.format_sequence = lambda x: pyjson.loads(x)
    elif isinstance(schema_object, str):
        schema = schema_object
        regex_str = build_regex_from_schema(schema)
        generator = regex(model, regex_str, max_tokens, sampler)
        generator.format_sequence = lambda x: pyjson.loads(x)
    else:
        raise ValueError(
            f"Cannot parse schema {schema_object}. The schema must be either "
            + "a Pydantic object, a function or a string that contains the JSON "
            + "Schema specification"
        )

    return generator


@json.register(Model)  # Register the Model class from mlx_lm.llama
def _(
    model,
    schema_object: Union[str, object, Callable],
    max_tokens: Optional[int] = None,
    sampler: Sampler = multinomial(),
):
    if isinstance(schema_object, type(BaseModel)):
        schema = pyjson.dumps(schema_object.model_json_schema())
        regex_str = build_regex_from_schema(schema, whitespace_pattern)
        generator = regex(model, regex_str, sampler)
        generator.format_sequence = lambda x: schema_object.parse_raw(x)
    elif callable(schema_object):
        schema = pyjson.dumps(get_schema_from_signature(schema_object))
        regex_str = build_regex_from_schema(schema, whitespace_pattern)
        generator = regex(model, regex_str, sampler)
        generator.format_sequence = lambda x: pyjson.loads(x)
    elif isinstance(schema_object, str):
        schema = schema_object
        regex_str = build_regex_from_schema(schema, whitespace_pattern)
        generator = regex(model, regex_str, sampler)
        generator.format_sequence = lambda x: pyjson.loads(x)
    else:
        raise ValueError(
            f"Cannot parse schema {schema_object}. The schema must be either "
            + "a Pydantic object, a function or a string that contains the JSON "
            + "Schema specification"
        )

    return generator
