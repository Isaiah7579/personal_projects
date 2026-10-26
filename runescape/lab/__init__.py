"""GE strategy lab: a GPU referee that scores any trading policy the same honest way, and evolves new ones.

Modules, in dependency order:
    market    - the 5m archive as dense [bucket, item] arrays (plus the backtester's ItemSeries)
    features  - point-in-time features at every check time
    splits    - rolling week-based train / validation / holdout splits
    env       - the batched simulator (fills, slots, buy limits, tax)
    policies  - the generic policy interface and the policy families
    score     - report cards
    evolve    - evolution strategies, writing runs the dashboard reads
"""
