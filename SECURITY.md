# Security policy

## Supported version

Security fixes target the latest `1.x` release and the current default branch.

## Report a problem

Use GitHub's private vulnerability reporting for this repository. Do not open a public issue for
a path traversal, command execution, stored script injection, unsafe file parsing, or secret leak
until a fix is available. Include the affected version, operating system, input type, smallest
reproduction, and expected impact. Do not include private documents or credentials.

## Product boundary

ContextBench binds to `127.0.0.1` by default and has no authentication. It is a single-user local
tool, not an Internet-facing multi-user service. Do not bind it to a public interface without an
authenticated reverse proxy, request limits, strict allowed hosts, and a separate review.

The built-in server accepts only loopback Host headers to reduce DNS-rebinding risk.

The application does not execute ingested code. It treats PDF and text content as data, caps
uploads, PDF page count, and extracted text size, generates server-side identifiers, and does not
serve uploaded files as active content.
Optional Ollama prompts separate system instructions, user questions, and quoted evidence.

Keep FastAPI, Starlette, `python-multipart`, pypdf, Qdrant, and frontend packages patched. File
parsing and multipart handling are security-sensitive dependency paths.
