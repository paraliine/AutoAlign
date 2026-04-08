"""Regression tests for the chatml-with-empty-think template fix.

The template used to inject the empty `<think>\\n\\n</think>\\n\\n` block
into `role_ends[HUMAN]`, producing the wrong token order
(`...<|im_end|>\\n<think>...</think>\\n\\n<|im_start|>assistant\\n...`)
that did not match the official Qwen3 chat template. After the fix the
template uses Qwen3NoThinkStrategy, which puts the empty think block
*inside* the assistant turn, after the `<|im_start|>assistant\\n` header,
matching the official jinja's `enable_thinking=false` branch.
"""
from autoalign.conversation import (
    Conversation,
    Qwen3NoThinkStrategy,
    Role,
    TEMPLATES,
)


THINK_BLOCK = "<think>\n\n</think>\n\n"
TEMPLATE_KEY = "chatml-with-empty-think"


# ----- template structure / regression checks -----


def test_template_strategy_is_qwen3_no_think():
    template = TEMPLATES[TEMPLATE_KEY]
    assert isinstance(template.strategy, Qwen3NoThinkStrategy)
    assert template.strategy.no_think_prefix == THINK_BLOCK


def test_template_name_matches_dict_key():
    """Regression: the internal `name=` was previously the wrong string
    `chatml-keep-system` due to a copy-paste from the keep-system variant."""
    assert TEMPLATES[TEMPLATE_KEY].name == TEMPLATE_KEY


def test_role_ends_human_no_longer_contains_think_block():
    """Regression: the buggy version embedded `<think>...</think>` in
    `role_ends[HUMAN]` so the block was emitted between the user's
    `<|im_end|>` and the next `<|im_start|>assistant`."""
    assert TEMPLATES[TEMPLATE_KEY].role_ends[Role.HUMAN] == "<|im_end|>\n"


# ----- rendering behavior -----


def _make_conv(messages):
    """Build a Conversation with the given messages, dropping the
    auto-injected empty SYSTEM turn so test expectations stay focused on
    the user/assistant interleaving."""
    conv = Conversation.from_template(TEMPLATE_KEY)
    conv.clear_message()
    for role, msg in messages:
        conv.append_message(role, msg)
    return conv


def test_single_turn_think_block_is_inside_assistant_turn():
    conv = _make_conv([(Role.HUMAN, "u1"), (Role.ASSISTANT, "a1")])
    rendered = conv.get_conversation_str()
    expected = (
        "<|im_start|>user\nu1<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\na1<|im_end|>\n"
    )
    assert rendered == expected


def test_single_turn_think_block_not_in_user_assistant_gap():
    """Regression for the buggy token order
    `...<|im_end|>\\n<think>...</think>\\n\\n<|im_start|>assistant\\n...`."""
    conv = _make_conv([(Role.HUMAN, "u1"), (Role.ASSISTANT, "a1")])
    rendered = conv.get_conversation_str()
    bad = "<|im_end|>\n<think>\n\n</think>\n\n<|im_start|>assistant"
    assert bad not in rendered


def test_multi_turn_only_last_assistant_gets_think_block():
    """Historical assistant turns must NOT carry the think prefix; only
    the assistant turn that follows the last real user query gets it."""
    conv = _make_conv(
        [
            (Role.HUMAN, "u1"),
            (Role.ASSISTANT, "a1"),
            (Role.HUMAN, "u2"),
            (Role.ASSISTANT, "a2"),
        ]
    )
    rendered = conv.get_conversation_str()
    assert "<|im_start|>assistant\na1<|im_end|>\n" in rendered
    assert (
        "<|im_start|>assistant\n<think>\n\n</think>\n\na2<|im_end|>\n" in rendered
    )
    assert rendered.count(THINK_BLOCK) == 1


def test_add_generation_prompt_appends_think_after_assistant_header():
    conv = _make_conv([(Role.HUMAN, "u1")])
    rendered = conv.get_conversation_str(add_generation_prompt=True)
    assert rendered.endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def test_multi_turn_tool_call_only_last_assistant_gets_prefix():
    """Qwen3-8B rule: in a multi-turn tool-calling conversation, only the
    very last message receives the empty think prefix. Intermediate
    assistant turns -- including those that emit `<tool_call>` and are
    followed by `<tool_response>` user messages -- get NOTHING, even
    though they sit after the last "real" user query.

    This is the key behavioral difference from Qwen3.5's chat template,
    which would attach the prefix to every assistant turn past the last
    real user query."""
    conv = _make_conv(
        [
            (Role.HUMAN, "u1"),
            (Role.ASSISTANT, "a1 (with tool_call)"),
            (Role.HUMAN, "<tool_response>result</tool_response>"),
            (Role.ASSISTANT, "a2"),
        ]
    )
    rendered = conv.get_conversation_str()
    # Intermediate tool-call assistant must NOT have the prefix.
    assert "<|im_start|>assistant\na1 (with tool_call)<|im_end|>\n" in rendered
    # Only the final assistant gets the prefix.
    assert (
        "<|im_start|>assistant\n<think>\n\n</think>\n\na2<|im_end|>\n" in rendered
    )
    # Exactly one think block in the entire output.
    assert rendered.count(THINK_BLOCK) == 1


def test_long_tool_call_chain_only_last_assistant_gets_prefix():
    """Mirrors the shape of real-world Qwen3-8B SFT data: many
    intermediate assistant tool_call turns interleaved with
    `<tool_response>` user messages, ending in a final natural-language
    assistant turn. Only that final turn must carry the empty think
    prefix; every intermediate assistant turn must stay clean."""
    conv = _make_conv(
        [
            (Role.HUMAN, "task description"),
            (Role.ASSISTANT, "I'll start by..."),
            (Role.ASSISTANT, "<tool_call>TodoWrite</tool_call>"),
            (Role.HUMAN, "<tool_response>todo created</tool_response>"),
            (Role.ASSISTANT, "<tool_call>Bash</tool_call>"),
            (Role.ASSISTANT, "<tool_call>Write</tool_call>"),
            (Role.HUMAN, "<tool_response>file created</tool_response>"),
            (Role.ASSISTANT, "Done."),
        ]
    )
    rendered = conv.get_conversation_str()
    # Exactly one think block in the entire output -- only on the last assistant.
    assert rendered.count(THINK_BLOCK) == 1
    # The last assistant's message carries the prefix.
    assert (
        "<|im_start|>assistant\n<think>\n\n</think>\n\nDone.<|im_end|>\n" in rendered
    )
    # All intermediate assistant turns stay clean.
    assert "<|im_start|>assistant\nI'll start by...<|im_end|>\n" in rendered
    assert (
        "<|im_start|>assistant\n<tool_call>TodoWrite</tool_call><|im_end|>\n"
        in rendered
    )
    assert (
        "<|im_start|>assistant\n<tool_call>Bash</tool_call><|im_end|>\n" in rendered
    )
    assert (
        "<|im_start|>assistant\n<tool_call>Write</tool_call><|im_end|>\n" in rendered
    )
