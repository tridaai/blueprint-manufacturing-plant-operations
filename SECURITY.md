# Security

## Reporting

Report vulnerabilities privately to the maintainers via the Trida AI
contact channels (https://trida.ai) rather than a public issue.

## Scope notes for this blueprint

- The agent in this repo **never actuates anything**: it produces
  draft work orders and escalation notes for a human supervisor. There
  is no machine-control path to secure — keep it that way in forks.
- Free-text fields on a case (notes, descriptions) are treated as
  untrusted input and pass an injection screen before any model reads
  them; safety-related instruction tampering ("skip the lockout") is
  flagged on the record, never obeyed.
- API keys and provider credentials come from the environment only
  (see `.env.example`). Never commit a real key; the test suite scrubs
  provider variables so a leaked shell key cannot turn a unit test
  into a live call.
- The optional `API_KEY` shared-secret gate is a demo-grade control,
  not an identity system; a deployment puts real auth in front.
