import uuid
import os
import glob
import yaml

from typing import List
from metadata import MetadataStore
from data_types import Preset
from models.pydantic_models import HumanModel, PersonaModel
from utils import list_human_files, list_persona_files
from constants import MEMGPT_DIR, DEFAULT_HUMAN, DEFAULT_PERSONA
from utils import get_human_text, get_persona_text
from functions.functions import load_all_function_sets
from prompts import gpt_system


def load_yaml_file(file_path):
    """
    Load a YAML file and return the data.

    :param file_path: Path to the YAML file.
    :return: Data from the YAML file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_all_presets():
    """Load all the preset configs in the examples directory"""

    ## Load the examples
    # Get the directory in which the script is located
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Construct the path pattern
    example_path_pattern = os.path.join(script_directory, "examples", "*.yaml")
    # Listing all YAML files
    example_yaml_files = glob.glob(example_path_pattern)

    ## Load the user-provided presets
    # ~/.memgpt/presets/*.yaml
    user_presets_dir = os.path.join(MEMGPT_DIR, "presets")
    # Create directory if it doesn't exist
    if not os.path.exists(user_presets_dir):
        os.makedirs(user_presets_dir)
    # Construct the path pattern
    user_path_pattern = os.path.join(user_presets_dir, "*.yaml")
    # Listing all YAML files
    user_yaml_files = glob.glob(user_path_pattern)

    # Pull from both examples and user-provided
    all_yaml_files = example_yaml_files + user_yaml_files

    # Loading and creating a mapping from file name to YAML data
    all_yaml_data = {}
    for file_path in all_yaml_files:
        # Extracting the base file name without the '.yaml' extension
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        data = load_yaml_file(file_path)
        all_yaml_data[base_name] = data

    return all_yaml_data


available_presets = load_all_presets()
preset_options = list(available_presets.keys())


def add_default_humans_and_personas(user_id: uuid.UUID, ms: MetadataStore):
    for persona_file in list_persona_files():
        text = open(persona_file, "r").read()
        name = os.path.basename(persona_file).replace(".txt", "")
        if ms.get_persona(user_id=user_id, name=name) is not None:
            continue
        persona = PersonaModel(name=name, text=text, user_id=user_id)
        ms.add_persona(persona)
    for human_file in list_human_files():
        text = open(human_file, "r").read()
        name = os.path.basename(human_file).replace(".txt", "")
        if ms.get_human(user_id=user_id, name=name) is not None:
            continue
        human = HumanModel(name=name, text=text, user_id=user_id)
        ms.add_human(human)


def generate_functions_json(preset_functions: List[str]):
    """
    Generate JSON schema for the functions based on what is locally available.

    TODO: store function definitions in the DB, instead of locally
    """
    # Available functions is a mapping from:
    # function_name -> {
    #   json_schema: schema
    #   python_function: function
    # }
    available_functions = load_all_function_sets()
    # Filter down the function set based on what the preset requested
    preset_function_set = {}
    for f_name in preset_functions:
        if f_name not in available_functions:
            raise ValueError(
                f"Function '{f_name}' was specified in preset, but is not in function library:\n{available_functions.keys()}"
            )
        preset_function_set[f_name] = available_functions[f_name]
    assert len(preset_functions) == len(preset_function_set)
    preset_function_set_schemas = [
        f_dict["json_schema"] for _, f_dict in preset_function_set.items()
    ]
    return preset_function_set_schemas


def load_preset(preset_name: str, user_id: uuid.UUID):
    preset_config = available_presets[preset_name]
    preset_system_prompt = preset_config["system_prompt"]
    preset_function_set_names = preset_config["functions"]
    functions_schema = generate_functions_json(preset_function_set_names)

    preset = Preset(
        user_id=user_id,
        name=preset_name,
        system=gpt_system.get_system_text(preset_system_prompt),
        persona=get_persona_text(DEFAULT_PERSONA),
        persona_name=DEFAULT_PERSONA,
        human=get_human_text(DEFAULT_HUMAN),
        human_name=DEFAULT_HUMAN,
        functions_schema=functions_schema,
    )
    return preset


def add_default_presets(user_id: uuid.UUID, ms: MetadataStore):
    """Add the default presets to the metadata store"""
    # make sure humans/personas added
    add_default_humans_and_personas(user_id=user_id, ms=ms)

    # add default presets
    for preset_name in preset_options:
        if ms.get_preset(user_id=user_id, name=preset_name) is not None:
            continue

        preset = load_preset(preset_name, user_id)
        ms.create_preset(preset)
