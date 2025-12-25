# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please:

1. **Do NOT** open a public issue
2. Contact the maintainer directly via Discord: mondeep.blend
3. Provide details of the vulnerability
4. Allow reasonable time for a fix before public disclosure

## Security Considerations

This project is a research prototype for ARC-AGI solving. It:
- Does not handle sensitive user data
- Does not make network requests to external services
- Runs entirely locally on your machine

**Note**: Always review code before running it, especially:
- Training scripts that may use significant compute
- Any code that reads/writes files
