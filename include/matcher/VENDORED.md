These six headers (`order_types.hpp`, `order_pool.hpp`, `order_index.hpp`,
`price_bitmap.hpp`, `intrusive_book.hpp`, `seqlock_snapshot.hpp`) are
vendored byte-for-byte from
[`lock-free-order-matching-engine`](https://github.com/Manas103/lock-free-order-matching-engine)
at commit `10d0d6b803cd8500e1486dd2faebcab8d69f58e3`, unmodified.

This project does not use that repo's gateway rings, seqlock market-data
publisher, or multi-threaded matching loop: a market-making simulation
replays one deterministic session at a time on a single thread, so the
concurrency machinery around the book has nothing to do here. What is reused
is the book itself, the thing whose correctness is hardest to get right
(price-time priority, O(1) cancel/amend, the conservation-checked
accounting). Submitting orders to a real matching core rather than an
assumed fill model is the reason this project extends that repo instead of
writing a toy book from scratch.

See the sibling repo's own README for the validation (reference-matcher diff
over 1M events, TSan/ASan clean) that these headers already carry.
