# SPDX-FileCopyrightText: 2024 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0

import re
from typing import Any, Dict, List, Union

"""
Utility functions for the Google Colab tutorial
"""


def update_config_variable(config_path: str, variable_path: str, new_value: Any) -> None:
    """
    Updates a specific variable in a Python configuration file while preserving structure.
    Can handle non-literal values like class references and nested dictionaries.
    Args:
        config_path: Path to the configuration file
        variable_path: Path to the variable using dictionary notation (e.g., 'DATASET["class"]')
        new_value: The new value to assign
    """
    with open(config_path, 'r') as file:
        lines = file.readlines()

    # Parse the variable path
    parts = variable_path.split('[')
    base_var = parts[0]
    keys = [p.split(']')[0].strip('"\'') for p in parts[1:]]

    # Convert the new value to a string representation
    if isinstance(new_value, (str, int, float, bool, list, tuple, dict, type(None))):
        if isinstance(new_value, str):
            value_str = f'"{new_value}"'
        else:
            value_str = repr(new_value)
    else:
        # For class references and other non-literal objects
        value_str = str(new_value)

    # Find the base variable and its bounds in the file
    base_var_start = -1
    base_var_end = -1
    in_base_var = False
    bracket_count = 0
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not in_base_var and (stripped.startswith(f"{base_var} =") or stripped.startswith(f"{base_var}=")):
            base_var_start = i
            in_base_var = True
            bracket_count += line.count('{') - line.count('}')
            # If the variable is defined on a single line
            if bracket_count == 0 and '{' in line and '}' in line:
                base_var_end = i
                break
        elif in_base_var:
            bracket_count += line.count('{') - line.count('}')
            if bracket_count == 0:
                base_var_end = i
                break

    if base_var_start == -1:
        raise ValueError(f"Variable {base_var} not found in the configuration file.")

    # If we're updating the base variable directly (no keys)
    if not keys:
        indentation = len(lines[base_var_start]) - len(lines[base_var_start].lstrip())
        lines[base_var_start] = ' ' * indentation + f"{base_var} = {value_str}\n"
        with open(config_path, 'w') as file:
            file.writelines(lines)
        return

    # Merge the lines into a single string for the variable
    var_content = ''.join(lines[base_var_start:base_var_end + 1])
    # Determine indentation of the base variable
    base_indent = len(lines[base_var_start]) - len(lines[base_var_start].lstrip())

    def update_nested(content: str, current_keys: List[str], value: str, indent: int) -> str:
        if not current_keys:
            return value

        key = current_keys[0]
        pattern = re.compile(rf'["\']?{re.escape(key)}["\']?\s*:\s*({{.*?}}|[^,\n}}]*)', re.DOTALL)
        match = pattern.search(content)

        if match:
            if len(current_keys) == 1:
                updated_value = value
            else:
                updated_value = update_nested(match.group(1), current_keys[1:], value, indent + 4)

            updated_content = content[:match.start(1)] + updated_value + content[match.end(1):]
            return updated_content
        else:
            # Key not found, add it
            closing_brace_index = content.rfind('}')
            if closing_brace_index > 0:
                needs_comma = not content[:closing_brace_index].strip().endswith(',')
                separator = ",\n" if needs_comma else "\n"
                updated_content = (
                    content[:closing_brace_index].rstrip() +
                    separator +
                    ' ' * (indent + 4) +
                    f'"{key}": {update_nested("{}", current_keys[1:], value, indent + 4)}\n' +
                    ' ' * indent +
                    content[closing_brace_index:]
                )
                return updated_content
            else:
                return f'"{key}": {update_nested("{}", current_keys[1:], value, indent + 4)}'

    if keys:
        updated_content = update_nested(var_content, keys, value_str, base_indent)
    else:
        updated_content = value_str

    # Split back into lines with proper indentation
    updated_lines = updated_content.split('\n')
    lines[base_var_start:base_var_end + 1] = [line + '\n' for line in updated_lines[:-1]] + [updated_lines[-1]]

    # Write the updated lines back to the file
    with open(config_path, 'w') as file:
        file.writelines(lines)

def update_class_reference(config_path: str, var_path: str, class_name: str, import_statement: str = None) -> None:
    """
    Updates a class reference in the configuration file and adds import if needed.

    Args:
        config_path: Path to the configuration file
        var_path: Path to the variable using dictionary notation (e.g., 'DATASET["class"]')
        class_name: Name of the class to reference
        import_statement: Optional import statement to add if not already present
    """
    # First check if the import exists (if provided)
    if import_statement:
        with open(config_path, 'r') as file:
            content = file.read()

        # Check if import already exists
        import_exists = False
        for line in content.split('\n'):
            if line.strip() == import_statement:
                import_exists = True
                break

        # Add import if needed
        if not import_exists:
            # Find where to insert the import
            lines = content.split('\n')
            last_import_line = -1

            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    last_import_line = i

            if last_import_line >= 0:
                lines.insert(last_import_line + 1, import_statement)
            else:
                # No imports found, add after the docstring
                for i, line in enumerate(lines):
                    if i > 0 and lines[i-1].startswith('"""') and not line.startswith('"""'):
                        lines.insert(i, import_statement)
                        break
                else:
                    # Just add at the top after any license comments
                    for i, line in enumerate(lines):
                        if not line.strip().startswith('#') and line.strip():
                            lines.insert(i, import_statement)
                            break

            with open(config_path, 'w') as file:
                file.write('\n'.join(lines))

def add_class_to_config(config_path, class_name, class_definition):
    """
    Adds a class definition to the configuration file.
    """
    with open(config_path, 'r') as file:
        content = file.read()

    class_pattern = re.compile(rf'class {class_name}\(.*?\).*?\n(.*?)(?=\n\n|\Z)', re.DOTALL)
    match = class_pattern.search(content)

    if match:
        content = content[:match.start()] + class_definition + content[match.end():]
    else:
        imports_end = re.search(r'^(from\s+\S+\s+import\s+\S+\s*\n)+\n', content, re.MULTILINE)
        if imports_end:
            insert_pos = imports_end.end()
            content = content[:insert_pos] + '\n' + class_definition + '\n' + content[insert_pos:]
        else:
            docstring_end = re.search(r'\"\"\".*?\"\"\"', content, re.DOTALL)
            if docstring_end:
                insert_pos = docstring_end.end()
                content = content[:insert_pos] + '\n\n' + class_definition + '\n' + content[insert_pos:]
            else:
                content = class_definition + '\n\n' + content

    with open(config_path, 'w') as file:
        file.write(content)

def initialize_config_file(config_path):
    """
    Initializes a new configuration file with basic structure and licenses.
    This configuration file upload synthetic data through a 

    Args:
        config_path (str): Path to the configuration file.
    """
    # Define the initial content of the configuration file
    initial_content = """
# SPDX-FileCopyrightText: 2024 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0

\"\"\"
Configuration file for Synthetic Images Metrics Toolkit.
Defines metrics, real data, and synthetic data configurations.
\"\"\"
from dataset import BaseDataset

METRICS = []

CONFIGS = {
    "RUN_DIR": "",
    "NUM_GPUS": 0,
    "VERBOSE": True,
    "OC_DETECTOR_PATH": None
}

METRICS_CONFIGS = {
    "nhood_size":
        {
            "pr": 5,
            "prdc": 5,
            "pr_auth": 5
        },
    "K-NN_configs":
        {
            "num_real": 3,
            "num_synth": 5
        },
    "padding": False
}

DATASET = {
    "class": CustomDataset,
    "params": {
        "path_data": None,
        "path_labels": None,
        "use_labels": False,
        "size_dataset": None
    }
}

# Flag to determine the mode of operation
USE_PRETRAINED_MODEL = False

SYNTHETIC_DATA = {

    # Configuration for direct synthetic images mode
    "from_files":
        {
        "class": CustomDataset,

        "params":
            {
            "path_data": "data/real_images_simulation.nii.gz",
            "path_labels": None,
            "use_labels": False,
            "size_dataset": None,
            }
        }
}
    """

    # Write the initial content to the file
    with open(config_path, 'w') as file:
        file.write(initial_content)

def initialize_config_file_gen(config_path):
    """
    Initializes a new configuration file with basic structure and licenses.
    This configuration file exploits a pre-trained generator to generate synthetic data.

    Args:
        config_path (str): Path to the configuration file.
    """
    # Define the initial content of the configuration file
    initial_content = """
# SPDX-FileCopyrightText: 2024 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0

\"\"\"
Configuration file for Synthetic Images Metrics Toolkit.
Defines metrics, real data, and synthetic data configurations.
\"\"\"
from dataset import BaseDataset

METRICS = []

CONFIGS = {
    "RUN_DIR": "",
    "NUM_GPUS": 0,
    "VERBOSE": True,
    "OC_DETECTOR_PATH": None
}

METRICS_CONFIGS = {
    "nhood_size":
        {
            "pr": 5,
            "prdc": 5,
            "pr_auth": 5
        },
    "K-NN_configs":
        {
            "num_real": 3,
            "num_synth": 5
        },
    "padding": False
}

DATASET = {
    "class": CustomDataset,
    "params": {
        "path_data": None,
        "path_labels": None,
        "use_labels": False,
        "size_dataset": None
    }
}

# Flag to determine the mode of operation
USE_PRETRAINED_MODEL = True

SYNTHETIC_DATA = {

    # Configuration for pre-trained model mode
    "pretrained_model":
        {
        "network_path": "",
        "load_network": load_network,
        "run_generator": run_generator,
        "NUM_SYNTH": 500
        },
}
    """

    # Write the initial content to the file
    with open(config_path, 'w') as file:
        file.write(initial_content)