import re


def find_mcq_end(string):
    patterns = [
        r'\([A-D]\)',
        r'[A-D][\.\)]',
        r'[a-d][\.\)]',
    ]

    combined_pattern = '|'.join(patterns)
    matches = list(re.finditer(combined_pattern, string, re.IGNORECASE))

    if len(matches) >= 4:
        last_match = matches[-1]
        return True, last_match.end()

    return False, -1


def find_next_newline(string):
    single_newline = string.find('\n')
    double_newline = string.find('\n\n')

    if single_newline == -1 and double_newline == -1:
        return -1
    elif single_newline == -1:
        return double_newline
    elif double_newline == -1:
        return single_newline
    else:
        return min(single_newline, double_newline)


def generate_variants(prefixes):
    variants = []
    for word in prefixes:
        clean_word = word.lstrip('#* ').rstrip(':')
        variants.append(f"{clean_word}:")
        variants.append(f"**{clean_word}**")
        variants.append(f"## {clean_word}:")

    return variants


def remove_prefix(string):
    # Remove numbers and alphabets with periods or brackets at the beginning
    string = re.sub(r'^(\d+\.|\w+\))\s*', '', string).strip()

    # Remove prefixes like "Task:", "Prompt:", "Question:"
    prefixes = [
        "Task", "Prompt", "Original Prompt", "Question", "The Problem", "Problem",
        "Scenario", "The senario", "Situation", "The situation", "Context",
        "Challenge", "Query", "Request", "Instructions", "Instruction",
        "Descriptions", "Description"
    ]
    prefixes = generate_variants(prefixes)

    prefix_patterns = [re.escape(prefix) for prefix in prefixes]

    prefix_patterns.append(r"Question \d+:")  # Question 1:, Question 2:, etc.
    prefix_patterns.append(r"Question \d+\.")  # Question 1., Question 2., etc.
    prefix_patterns.append(r"Part \d+:")  # Part 1:, Part 2:, etc.
    prefix_patterns.append(r"Q\d+\.")  # Q1., Q2., etc.
    prefix_patterns.append(r"Q\d+:")  # Q1:, Q2:, etc.

    combined_pattern = "|".join(prefix_patterns)

    match = re.match(f"^({combined_pattern})\s*", string, re.IGNORECASE)
    if match:
        return string[match.end():].strip()

    return string


def extract_final_answer(text, model_path=""):
    """Extract the final answer from a reasoning model's output.

    Qwen3-style reasoning models produce outputs of the form:
        ' thinking\n<reasoning> response\n\n<final answer>'
    This strips the leading ' thinking' marker and everything up to and
    including the final ' response' marker, returning only the final answer.
    For non-reasoning models (or when no reasoning markers are present) the
    text is returned unchanged.
    """
    if not text:
        return text
    # Only strip reasoning markers for Qwen3-style reasoning models.
    if "qwen3" not in model_path.lower():
        return text.strip()
    text = text.strip()
    # Take everything after the last ' response' marker (the actual answer).
    for marker in (" response\n\n", " response\n", " response"):
        idx = text.rfind(marker)
        if idx != -1:
            text = text[idx + len(marker):]
            break
    # If the model answered without a ' response' marker, drop a leading ' thinking'.
    if text.startswith("thinking\n"):
        text = text[len("thinking\n"):].lstrip("\n")
    return text.strip()


def instruction_post_process(instruction, model_path):
    if "gemma-2" in model_path.lower():
        # remove the prefix
        instruction = remove_prefix(instruction)
        # find mcq problems
        is_mcq, end_pos = find_mcq_end(instruction)
        assistant_markers = ["Answer:", "Answers:", "The answer is", "Correct answer", "The correct answer", "Answer is", "Explanation:", "Here are some", "Solution Approach:", "Solution:"]
        assistant_pattern = r'(?:' + '|'.join(assistant_markers) + ')s?:'
        assistant_match = re.search(assistant_pattern, instruction)  # Exact match, no re.IGNORECASE

        if instruction.startswith("*"):
            if '?' in instruction:
                instruction = instruction.split('?')[0].replace("*", "").strip() + '?'
                instruction = remove_prefix(instruction)
                class_num = 1
                return instruction, class_num

        instruction = remove_prefix(instruction)

        if instruction.startswith('\"'):
            if '?' in instruction:
                instruction = instruction.split('?')[0].replace("\"", "").strip() + '?'
                class_num = 2.1
            else:
                instruction = instruction.split('\n')[0].replace("\"", "").strip()
                instruction = instruction.replace("*", "").strip()
                class_num = 2.2
        elif instruction.startswith('<b>'):
            instruction.split('\n')[0].replace('</b>', "").replace('<b>', "").strip()
            instruction = instruction.replace("*", "").strip()
            class_num = 3
        elif assistant_match:
            instruction = instruction[:assistant_match.start()].strip()
            instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
            class_num = 4
        elif instruction.split('\n')[0].strip().endswith(':'):
            colon_pos = instruction.split('\n')[0].strip().rfind(':')
            if '#' in instruction:
                instruction = instruction.split('#')[0].strip()
                instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
                class_num = 5.1
            elif '?' in instruction:
                instruction = instruction.split('?')[0].strip() + '?'
                instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
                class_num = 5.2
            else:
                instruction = instruction.split('\n')[0].strip()
                instruction = instruction.replace("*", "").strip()
                class_num = 5.3
        else:
            if '?' in instruction:
                instruction = instruction.split('?')[0].strip() + '?'
                instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
                class_num = 6.1
            else:
                instruction = instruction.split('\n')[0].strip()
                instruction = instruction.replace("*", "").strip()
                class_num = 6.2

        # Remove prefixes again
        instruction = remove_prefix(instruction)

        return instruction, class_num

    elif "llama-3" in model_path.lower():
        # remove the prefix
        instruction = remove_prefix(instruction)
        # find mcq problems
        is_mcq, end_pos = find_mcq_end(instruction)
        assistant_markers = ["Answer:", "Answers:", "The answer is", "Correct answer", "The correct answer", "Answer is", "Explanation:", "Here are some", "Solution Approach:", "Solution:"]
        assistant_pattern = r'(?:' + '|'.join(assistant_markers) + ')s?:'
        assistant_match = re.search(assistant_pattern, instruction)  # Exact match, no re.IGNORECASE

        step_makers = ["# Step 1", "## Step 1", "### Step 1"]
        step_pattern = r'(?:' + '|'.join(step_makers) + r'):?'
        step_match = re.search(step_pattern, instruction)  # Exact match, no re.IGNORECASE

        if instruction.startswith("*"):
            if '?' in instruction:
                instruction = instruction.split('?')[0].replace("*", "").strip() + '?'
                instruction = remove_prefix(instruction)
                class_num = 1
                return instruction, class_num

        instruction = remove_prefix(instruction)
        if instruction.startswith('\"'):
            if '?' in instruction:
                instruction = instruction.split('?')[0].replace("\"", "").strip() + '?'
                class_num = 2.1
            else:
                instruction = instruction.split('\n')[0].replace("\"", "").strip()
                instruction = instruction.replace("*", "").strip()
                class_num = 2.2
        elif instruction.startswith('<b>'):
            instruction.split('\n')[0].replace('</b>', "").replace('<b>', "").strip()
            instruction = instruction.replace("*", "").strip()
            class_num = 3
        elif assistant_match:
            instruction = instruction[:assistant_match.start()].strip()
            instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
            class_num = 4
        elif step_match:
            instruction = instruction[:step_match.start()].strip()
            instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
            class_num = 5
        elif instruction.split('\n')[0].strip().endswith(':'):
            colon_pos = instruction.split('\n')[0].strip().rfind(':')
            if '#' in instruction:
                instruction = instruction.split('#')[0].strip()
                instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
                class_num = 6.1
            elif '?' in instruction:
                instruction = instruction.split('?')[0].strip() + '?'
                instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
                class_num = 6.2
            else:
                instruction = instruction.split('\n')[0].strip()
                instruction = instruction.replace("*", "").strip()
                class_num = 6.3
        else:
            if '?' in instruction:
                instruction = instruction.split('?')[0].strip() + '?'
                instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
                class_num = 99.1
            else:
                instruction = instruction.split('\n')[0].strip()
                instruction = instruction.replace("*", "").strip()
                class_num = 99.2

        # Remove prefixes again
        instruction = remove_prefix(instruction)

        return instruction, class_num

    else:
        return instruction, 0
