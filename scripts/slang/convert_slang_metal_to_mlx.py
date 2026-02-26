#!/usr/bin/env python3

import argparse
import json
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Parameter:
    raw: str
    name: str
    type_text: str
    attrs: tuple[str, ...]
    buffer_index: int | None


@dataclass(frozen=True)
class KernelFunction:
    name: str
    params: tuple[Parameter, ...]
    body: str
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Slang-generated Metal kernel into an MLXFast.metalKernel Swift snippet."
        )
    )
    parser.add_argument(
        "--metal",
        type=Path,
        required=True,
        help="Path to Slang-generated .metal source.",
    )
    parser.add_argument(
        "--entry",
        default=None,
        help="Kernel entry name. Defaults to the first [[kernel]] function.",
    )
    parser.add_argument(
        "--swift-out",
        type=Path,
        required=True,
        help="Output path for generated Swift snippet.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional output path for extracted metadata JSON.",
    )
    parser.add_argument(
        "--kernel-var",
        default=None,
        help="Swift variable name for MLXFast kernel. Default: <entry>Kernel",
    )
    parser.add_argument(
        "--kernel-name",
        default=None,
        help="MLX kernel name. Default: <entry>.",
    )
    parser.add_argument(
        "--input-names",
        default=None,
        help="Comma-separated input buffer names override.",
    )
    parser.add_argument(
        "--output-names",
        default=None,
        help="Comma-separated output buffer names override.",
    )
    parser.add_argument(
        "--keep-line-directives",
        action="store_true",
        help="Keep #line directives from Slang output.",
    )
    return parser.parse_args()


def split_top_level(text: str, delimiter: str) -> list[str]:
    parts: list[str] = []
    start = 0
    round_depth = 0
    square_depth = 0
    angle_depth = 0
    curly_depth = 0

    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "(":
            round_depth += 1
        elif ch == ")":
            round_depth = max(0, round_depth - 1)
        elif ch == "[":
            square_depth += 1
        elif ch == "]":
            square_depth = max(0, square_depth - 1)
        elif ch == "<":
            angle_depth += 1
        elif ch == ">":
            angle_depth = max(0, angle_depth - 1)
        elif ch == "{":
            curly_depth += 1
        elif ch == "}":
            curly_depth = max(0, curly_depth - 1)
        elif (
            ch == delimiter
            and round_depth == 0
            and square_depth == 0
            and angle_depth == 0
            and curly_depth == 0
        ):
            parts.append(text[start:i])
            start = i + 1
        i += 1

    parts.append(text[start:])
    return parts


def find_matching(text: str, start: int, open_ch: str, close_ch: str) -> int:
    if text[start] != open_ch:
        raise ValueError(f"Expected '{open_ch}' at index {start}")

    depth = 1
    i = start + 1
    in_string = False
    string_quote = ""
    in_line_comment = False
    in_block_comment = False

    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == string_quote:
                in_string = False
                string_quote = ""
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch in ("'", '"'):
            in_string = True
            string_quote = ch
            i += 1
            continue

        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1

    raise ValueError(f"Unmatched '{open_ch}' from index {start}")


def parse_parameter(param_text: str) -> Parameter:
    text = param_text.strip()
    if not text:
        raise ValueError("Empty parameter text")

    attrs = tuple(re.findall(r"\[\[\s*([a-zA-Z_][a-zA-Z0-9_]*)(?:\([^\]]*\))?\s*\]\]", text))
    buffer_match = re.search(r"\[\[\s*buffer\((\d+)\)\s*\]\]", text)
    buffer_index = int(buffer_match.group(1)) if buffer_match else None

    without_attrs = re.sub(r"\[\[[^\]]+\]\]", "", text).strip()
    name_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", without_attrs)
    if not name_match:
        raise ValueError(f"Failed to parse parameter name from: {text}")

    name = name_match.group(1)
    type_text = without_attrs[: name_match.start()].strip()
    if not type_text:
        raise ValueError(f"Failed to parse parameter type from: {text}")

    return Parameter(
        raw=text,
        name=name,
        type_text=type_text,
        attrs=attrs,
        buffer_index=buffer_index,
    )


def find_kernel_functions(source: str) -> list[KernelFunction]:
    kernels: list[KernelFunction] = []
    pattern = re.compile(r"\[\[kernel\]\]\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    pos = 0
    while True:
        match = pattern.search(source, pos)
        if not match:
            break
        name = match.group(1)
        open_paren = source.find("(", match.end() - 1)
        close_paren = find_matching(source, open_paren, "(", ")")
        params_text = source[open_paren + 1 : close_paren]

        brace_start = close_paren + 1
        while brace_start < len(source) and source[brace_start].isspace():
            brace_start += 1
        if brace_start >= len(source) or source[brace_start] != "{":
            raise ValueError(f"Expected '{{' for kernel {name}")
        brace_end = find_matching(source, brace_start, "{", "}")

        params: list[Parameter] = []
        for p in split_top_level(params_text, ","):
            stripped = p.strip()
            if stripped:
                params.append(parse_parameter(stripped))

        body = source[brace_start + 1 : brace_end]
        kernels.append(
            KernelFunction(
                name=name,
                params=tuple(params),
                body=body,
                start=match.start(),
                end=brace_end + 1,
            )
        )
        pos = brace_end + 1
    return kernels


def strip_line_directives(text: str) -> str:
    return re.sub(r"^\s*#line[^\n]*\n", "", text, flags=re.MULTILINE)


def strip_unused_kernel_context_assignments(body: str) -> str:
    lines = body.splitlines()
    if not lines:
        return body

    to_remove: set[int] = set()
    decl_re = re.compile(r"^\s*thread\s+KernelContext_[A-Za-z0-9_]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*;\s*$")

    for idx, line in enumerate(lines):
        decl_match = decl_re.match(line)
        if not decl_match:
            continue
        var_name = decl_match.group(1)
        var_token = re.compile(rf"\b{re.escape(var_name)}\b")
        assign_re = re.compile(
            rf"^\s*\(&\s*{re.escape(var_name)}\s*\)->[A-Za-z_][A-Za-z0-9_]*\s*=\s*[A-Za-z_][A-Za-z0-9_]*\s*;\s*$"
        )

        occurrences = [i for i, candidate in enumerate(lines) if var_token.search(candidate)]
        if not occurrences:
            continue

        safe_to_drop = True
        for occurrence in occurrences:
            if occurrence == idx:
                continue
            if not assign_re.match(lines[occurrence]):
                safe_to_drop = False
                break

        if not safe_to_drop:
            continue

        to_remove.add(idx)
        for occurrence in occurrences:
            if occurrence != idx:
                to_remove.add(occurrence)

    if not to_remove:
        return body

    stripped_lines = [line for i, line in enumerate(lines) if i not in to_remove]
    stripped = "\n".join(stripped_lines)
    return re.sub(r"\n{3,}", "\n\n", stripped)


def inline_kernel_context(body: str, header: str) -> tuple[str, str]:
    decl_re = re.compile(
        r"^\s*thread\s+(KernelContext_[A-Za-z0-9_]+)\s+([A-Za-z_][A-Za-z0-9_]*)\s*;\s*$",
        flags=re.MULTILINE,
    )
    contexts = decl_re.findall(body)
    if not contexts:
        return body, header

    body_text = body
    header_text = header

    for struct_name, var_name in contexts:
        assign_re = re.compile(
            rf"^\s*\(&\s*{re.escape(var_name)}\s*\)->([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*;\s*$",
            flags=re.MULTILINE,
        )
        assignments = assign_re.findall(body_text)
        mapping = {lhs: rhs for lhs, rhs in assignments}

        body_text = assign_re.sub("", body_text)
        body_text = re.sub(
            rf"^\s*thread\s+{re.escape(struct_name)}\s+{re.escape(var_name)}\s*;\s*$",
            "",
            body_text,
            flags=re.MULTILINE,
        )

        access_re = re.compile(
            rf"\(&\s*{re.escape(var_name)}\s*\)->([A-Za-z_][A-Za-z0-9_]*)"
        )

        def replace_access(match: re.Match[str]) -> str:
            field = match.group(1)
            return mapping.get(field, field)

        body_text = access_re.sub(replace_access, body_text)

        if re.search(rf"\(&\s*{re.escape(var_name)}\s*\)->", body_text):
            raise ValueError(
                f"Failed to inline all KernelContext accesses for variable '{var_name}'."
            )

        header_text = re.sub(
            rf"\bstruct\s+{re.escape(struct_name)}\s*\{{.*?\}};\s*",
            "",
            header_text,
            flags=re.DOTALL,
        )

    body_text = re.sub(r"\n{3,}", "\n\n", body_text).strip("\n")
    header_text = re.sub(r"\n{3,}", "\n\n", header_text).strip("\n")
    return body_text, header_text


def detect_written_buffers(body: str, param_name: str) -> bool:
    n = re.escape(param_name)
    patterns = [
        rf"\b{n}\s*\[[^\]]*\]\s*[-+*/%&|^]?=",
        rf"\*\s*\(\s*{n}\s*\+[^\)]*\)\s*[-+*/%&|^]?=",
        rf"\batomic_[a-zA-Z0-9_]*\s*\(\s*&\s*{n}\s*\[",
        rf"\batomic_[a-zA-Z0-9_]*\s*\(\s*{n}\s*,",
    ]
    return any(re.search(pattern, body) for pattern in patterns)


def choose_io_names(
    kernel: KernelFunction,
    body: str,
) -> tuple[list[str], list[str], list[dict[str, object]]]:
    buffer_params = sorted(
        [p for p in kernel.params if p.buffer_index is not None],
        key=lambda p: p.buffer_index if p.buffer_index is not None else -1,
    )
    if not buffer_params:
        raise ValueError("No [[buffer(n)]] parameters found in kernel signature.")

    inputs: list[str] = []
    outputs: list[str] = []
    details: list[dict[str, object]] = []

    for p in buffer_params:
        written = detect_written_buffers(body, p.name)
        if written:
            outputs.append(p.name)
            role = "output"
        else:
            inputs.append(p.name)
            role = "input"
        details.append(
            {
                "name": p.name,
                "buffer_index": p.buffer_index,
                "role": role,
                "type": p.type_text,
            }
        )

    if not outputs:
        # deterministic fallback: last buffer is output
        fallback = buffer_params[-1].name
        outputs.append(fallback)
        inputs = [p.name for p in buffer_params[:-1]]
        for item in details:
            item["role"] = "output" if item["name"] == fallback else "input"
        print(
            (
                "warning: could not infer outputs by write-pattern; "
                f"falling back to last buffer as output ({fallback})."
            ),
            file=sys.stderr,
        )

    return inputs, outputs, details


def build_attribute_aliases(kernel: KernelFunction) -> list[str]:
    aliases: list[str] = []
    for p in kernel.params:
        if p.buffer_index is not None:
            continue
        attr_name = next((a for a in p.attrs if a != "kernel"), None)
        if not attr_name:
            continue
        # MLX generates attribute arguments named exactly as the attribute token.
        if p.name == attr_name:
            continue
        aliases.append(f"{p.type_text} {p.name} = {attr_name};")
    return aliases


def to_swift_array_literal(values: list[str]) -> str:
    return "[" + ", ".join(json.dumps(v) for v in values) + "]"


def to_swift_multiline(text: str, indent: str) -> str:
    stripped = text.strip("\n")
    if not stripped:
        return indent + '#"""\n' + indent + '"""#'
    lines = stripped.splitlines()
    rendered = [indent + '#"""']
    rendered.extend(indent + line for line in lines)
    rendered.append(indent + '"""#')
    return "\n".join(rendered)


def parse_name_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


def metadata_path(path: Path, project_root: Path) -> str:
    if path.is_absolute():
        try:
            return str(path.relative_to(project_root))
        except ValueError:
            return path.name
    return str(path)


def generate_swift_snippet(
    kernel_name: str,
    kernel_var: str,
    input_names: list[str],
    output_names: list[str],
    header: str,
    source_body: str,
) -> str:
    source_literal = to_swift_multiline(source_body, indent="        ")
    lines = [
        "// Generated by scripts/slang/convert_slang_metal_to_mlx.py",
        "import MLX",
        "",
        f"let {kernel_var} = MLXFast.metalKernel(",
        f"    name: {json.dumps(kernel_name)},",
        f"    inputNames: {to_swift_array_literal(input_names)},",
        f"    outputNames: {to_swift_array_literal(output_names)},",
        "    source:",
        source_literal + ("," if header.strip() else ""),
    ]
    if header.strip():
        header_literal = to_swift_multiline(header, indent="        ")
        lines.extend(
            [
                "    header:",
                header_literal,
            ]
        )
    lines.append(")")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    metal_path = args.metal
    if not metal_path.is_absolute():
        metal_path = project_root / metal_path
    if not metal_path.exists():
        raise SystemExit(f"Metal file not found: {metal_path}")

    source = metal_path.read_text(encoding="utf-8")
    kernels = find_kernel_functions(source)
    if not kernels:
        raise SystemExit(f"No [[kernel]] function found in: {metal_path}")

    if args.entry:
        selected = next((k for k in kernels if k.name == args.entry), None)
        if selected is None:
            available = ", ".join(k.name for k in kernels)
            raise SystemExit(
                f"Entry '{args.entry}' not found. Available kernels: {available}"
            )
    else:
        selected = kernels[0]

    header_text = source[: selected.start]
    body_text = selected.body

    if not args.keep_line_directives:
        header_text = strip_line_directives(header_text)
        body_text = strip_line_directives(body_text)
    body_text = strip_unused_kernel_context_assignments(body_text)
    body_text, header_text = inline_kernel_context(body_text, header_text)

    explicit_inputs = parse_name_list(args.input_names)
    explicit_outputs = parse_name_list(args.output_names)

    if explicit_inputs and explicit_outputs:
        input_names = explicit_inputs
        output_names = explicit_outputs
        buffer_params = sorted(
            [p for p in selected.params if p.buffer_index is not None],
            key=lambda p: p.buffer_index if p.buffer_index is not None else -1,
        )
        io_details = []
        for p in buffer_params:
            role = "output" if p.name in output_names else "input"
            io_details.append(
                {
                    "name": p.name,
                    "buffer_index": p.buffer_index,
                    "role": role,
                    "type": p.type_text,
                }
            )
    else:
        input_names, output_names, io_details = choose_io_names(selected, body_text)
        if explicit_inputs:
            input_names = explicit_inputs
        if explicit_outputs:
            output_names = explicit_outputs
    aliases = build_attribute_aliases(selected)

    body_core = textwrap.dedent(body_text).strip("\n")
    if aliases:
        alias_block = "\n".join(aliases)
        source_body = alias_block + ("\n" + body_core if body_core else "")
    else:
        source_body = body_core

    kernel_name = args.kernel_name or selected.name
    kernel_var = args.kernel_var or f"{selected.name}Kernel"
    swift = generate_swift_snippet(
        kernel_name=kernel_name,
        kernel_var=kernel_var,
        input_names=input_names,
        output_names=output_names,
        header=header_text,
        source_body=source_body,
    )

    swift_out = args.swift_out
    if not swift_out.is_absolute():
        swift_out = project_root / swift_out
    swift_out.parent.mkdir(parents=True, exist_ok=True)
    swift_out.write_text(swift, encoding="utf-8")

    metadata = {
        "metal": metadata_path(metal_path, project_root),
        "entry": selected.name,
        "kernel_name": kernel_name,
        "kernel_var": kernel_var,
        "input_names": input_names,
        "output_names": output_names,
        "source": source_body,
        "header": header_text,
        "buffer_parameters": io_details,
        "attribute_aliases": aliases,
        "line_directives_kept": bool(args.keep_line_directives),
        "swift_out": metadata_path(swift_out, project_root),
    }

    if args.json_out:
        json_out = args.json_out
        if not json_out.is_absolute():
            json_out = project_root / json_out
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"converted: {metal_path}")
    print(f"entry: {selected.name}")
    print(f"inputs: {', '.join(input_names)}")
    print(f"outputs: {', '.join(output_names)}")
    print(f"swift: {swift_out}")


if __name__ == "__main__":
    main()
