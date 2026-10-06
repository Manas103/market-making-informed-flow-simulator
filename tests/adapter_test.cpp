// Tests the one piece of new adapter code this project writes against the
// vendored book (detail::submit_new): hand-crafted scenarios where the
// expected trades, fills and book invariants are known in advance.
#include <string>
#include <vector>

#include "check.hpp"
#include "session_runner.hpp"

using namespace mmsim;
using lfmatch::Side;
using lfmatch::Trade;

static void test_resting_and_cross() {
    OrderPool pool(1024);
    Book book(pool, 128);
    std::vector<Trade> out;

    // MM rests a bid at value 0 (price_idx = kCenterOffset - half_spread).
    detail::submit_new(book, pool, out, /*id=*/1, Side::Buy, -2, 10, false);
    CHECK(out.empty());

    // A market sell for qty 3 should hit the resting bid exactly.
    out.clear();
    detail::submit_new(book, pool, out, /*id=*/2, Side::Sell, 0, 3, true);
    CHECK_EQ(out.size(), size_t{1});
    CHECK_EQ(out[0].qty, uint32_t{3});
    CHECK_EQ(out[0].resting_order_id, uint64_t{1});
    CHECK_EQ(out[0].aggressor_order_id, uint64_t{2});

    std::string err;
    CHECK(book.check_invariants(err));
    CHECK_EQ(book.book_qty(), uint64_t{7});  // 10 - 3 left resting
}

static void test_discard_when_empty() {
    OrderPool pool(1024);
    Book book(pool, 128);
    std::vector<Trade> out;

    // No resting liquidity at all: a market buy must discard, not crash or
    // rest (market orders never rest; see IntrusiveBook::submit).
    detail::submit_new(book, pool, out, /*id=*/1, Side::Buy, 0, 5, true);
    CHECK(out.empty());

    std::string err;
    CHECK(book.check_invariants(err));
    CHECK_EQ(book.book_qty(), uint64_t{0});
    CHECK_EQ(book.discarded_qty(), uint64_t{5});
}

static void test_cancel_removes_quote() {
    OrderPool pool(1024);
    Book book(pool, 128);
    std::vector<Trade> out;

    detail::submit_new(book, pool, out, /*id=*/1, Side::Sell, 2, 4, false);
    CHECK_EQ(book.book_qty(), uint64_t{4});

    book.cancel(1);
    CHECK_EQ(book.book_qty(), uint64_t{0});

    // A taker that would have hit the cancelled ask now finds nothing.
    out.clear();
    detail::submit_new(book, pool, out, /*id=*/2, Side::Buy, 0, 1, true);
    CHECK(out.empty());
}

static void test_partial_fill_keeps_remainder_resting() {
    OrderPool pool(1024);
    Book book(pool, 128);
    std::vector<Trade> out;

    detail::submit_new(book, pool, out, /*id=*/1, Side::Buy, -1, 6, false);
    out.clear();
    detail::submit_new(book, pool, out, /*id=*/2, Side::Sell, 0, 2, true);
    CHECK_EQ(out.size(), size_t{1});
    CHECK_EQ(out[0].qty, uint32_t{2});
    CHECK_EQ(book.book_qty(), uint64_t{4});

    std::string err;
    CHECK(book.check_invariants(err));
}

int main() {
    test_resting_and_cross();
    test_discard_when_empty();
    test_cancel_removes_quote();
    test_partial_fill_keeps_remainder_resting();
    return check_summary("adapter_test");
}
