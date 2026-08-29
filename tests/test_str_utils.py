import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'exp')))

import str_utils


def test_qwen3_strips_reasoning():
    text = " thinking\nLet me think carefully.\n response\n\nHere is the final answer."
    assert str_utils.extract_final_answer(text, "Qwen/Qwen3.5-4B") == "Here is the final answer."


def test_deepseek_strips_reasoning():
    text = " thinking\nSome chain of thought.\n response\n\nFinal answer here."
    assert str_utils.extract_final_answer(text, "deepseek-v4-flash:0731-cloud") == "Final answer here."


def test_non_reasoning_model_unchanged():
    text = " thinking\nThis is literal text.\n response\n\nNot a marker."
    assert str_utils.extract_final_answer(text, "llama-3.3-70b") == text.strip()


def test_no_markers_unchanged():
    text = "Just a plain instruction without markers."
    assert str_utils.extract_final_answer(text, "Qwen/Qwen3.5-4B") == text


def test_empty_input():
    assert str_utils.extract_final_answer("", "Qwen/Qwen3.5-4B") == ""
    assert str_utils.extract_final_answer(None, "Qwen/Qwen3.5-4B") is None


def test_leading_thinking_without_response():
    text = "thinking\nOnly reasoning, no response marker."
    assert str_utils.extract_final_answer(text, "Qwen/Qwen3.5-4B") == "Only reasoning, no response marker."


def test_word_response_not_treated_as_marker():
    # The word 'response' inside the text must not be mistaken for a reasoning marker.
    text = " thinking\nThe user asked for a response marker test.\n response\n\nFinal answer."
    assert str_utils.extract_final_answer(text, "Qwen/Qwen3.5-4B") == "Final answer."


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All tests passed.")
