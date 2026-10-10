# Intentional omissions

This directory exists to explain Stage C's learning logic. It intentionally
does not include:

- simulator or RSL-RL runtime packages;
- command-line entry points or launch scripts;
- model or optimizer payloads;
- generated metrics, plots, videos, logs, or example results;
- filesystem-specific configuration;
- scheduler, upload, installation, or deployment machinery;
- provenance contracts, receipts, guards, audit tools, or recovery code;
- artifact serialization, persistence, or publication functions;
- production tests and fixtures.

Consequently, this export cannot reproduce or launch training by itself. The
omissions do not change the parameters written here; they remove operational
and reporting layers that are unnecessary for understanding the method.
