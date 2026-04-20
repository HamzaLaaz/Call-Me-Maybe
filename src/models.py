from pydantic import BaseModel
from enum import Enum
from typing import Literal, Dict, Any


class State(Enum):
    """All possible states during JSON generation."""
    START = "start"
    EXPECT_NAME_KEY = "expect_name_key"
    EXPECT_NAME_VALUE = "expect_name_value"
    GENERATING_NAME = "generating_name"
    EXPECT_PARAMS_KEY = "expect_params_key"
    EXPECT_PARAMS_OPEN = "expect_params_open"
    EXPECT_ARG_KEY = "expect_arg_key"
    EXPECT_ARG_VALUE = "expect_arg_value"
    GENERATING_STRING = "generating_string"
    GENERATING_NUMBER = "generating_number"
    AFTER_ARG = "after_arg"
    DONE = "done"


class FunctionParameter(BaseModel):
    """Defines the metadata/type of a parameter in the schema."""
    type: Literal["number", "string"]


class FunctionDefinition(BaseModel):
    """The schema provided in functions_definition.json."""
    name: str
    description: str
    parameters: Dict[str, FunctionParameter]
    returns: FunctionParameter


class Prompt(BaseModel):
    """The input structure from function_calling_tests.json."""
    prompt: str


class FunctionCallResult(BaseModel):
    """
    The final output structure.
    Note: parameters here stores the GENERATED values, not the types.
    """
    prompt: str
    name: str
    parameters: Dict[str, Any]
