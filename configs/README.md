# Configuration directory contract

The active CGR-v3.1/v3.2 implementation has no YAML configuration dependency;
its frozen parameters are recorded in `docs/current_status.md` and report
provenance.

`archive/` contains configurations for concluded candidate families. New active
configuration files may live at this top level only when an active model or
canonical evaluator actually loads them.
