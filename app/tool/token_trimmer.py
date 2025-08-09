from typing import Dict

from app.tool.base import BaseTool


class TokenTrimmer(BaseTool):
    """Trim unnecessary tokens from long text by applying simple reductions."""

    name: str = "token_trimmer"
    description: str = (
        "Trim long text by collapsing whitespace, deduplicating lines, optionally removing code blocks, and truncating to max_chars."
    )

    parameters: dict = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The input text to trim."},
            "max_chars": {
                "type": "integer",
                "description": "Maximum output characters after trimming.",
                "default": 8000,
            },
            "strip_code": {
                "type": "boolean",
                "description": "Remove fenced code blocks (```...```).",
                "default": False,
            },
            "deduplicate_lines": {
                "type": "boolean",
                "description": "Remove duplicate lines to reduce redundancy.",
                "default": True,
            },
            "collapse_whitespace": {
                "type": "boolean",
                "description": "Collapse multiple blank lines and spaces.",
                "default": True,
            },
        },
        "required": ["text"],
    }

    async def execute(
        self,
        text: str,
        max_chars: int = 8000,
        strip_code: bool = False,
        deduplicate_lines: bool = True,
        collapse_whitespace: bool = True,
    ) -> Dict:
        import re

        original_len = len(text or "")
        processed = text or ""

        if strip_code:
            processed = re.sub(r"```[\s\S]*?```", "\n", processed)

        if deduplicate_lines:
            seen = set()
            new_lines = []
            for line in processed.splitlines():
                if line in seen:
                    continue
                seen.add(line)
                new_lines.append(line)
            processed = "\n".join(new_lines)

        if collapse_whitespace:
            processed = re.sub(r"\n{3,}", "\n\n", processed)
            processed = re.sub(r"[ \t]{2,}", " ", processed)

        if len(processed) > max_chars:
            head = max_chars // 2
            tail = max_chars - head
            processed = processed[:head] + "\n...\n" + processed[-tail:]

        return {
            "observation": processed,
            "original_length": original_len,
            "final_length": len(processed),
            "reduction": max(0, original_len - len(processed)),
            "success": True,
        }