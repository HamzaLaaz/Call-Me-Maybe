from pydantic import BaseModel
from typing import Literal, Dict, Any


class FunctionParameter(BaseModel):
    """Defines the metadata/type of a parameter in the schema."""
    type: Literal["number", "string", "boolean", "integer"]


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
