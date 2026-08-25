# Storage engine notes

## Write-ahead logging

A database can make a transaction durable by writing its log record before it writes the
corresponding data page. Recovery reads the log after a crash. It repeats committed changes that
did not reach the main data file and removes changes from transactions that never committed.

The order matters. If a dirty page reaches disk before the log record that describes the change,
recovery has no durable evidence for that update. A write-ahead log therefore imposes a flush
rule: the log sequence number for a page change must be durable before the page itself can be
flushed.

## B-tree pages

A B-tree keeps sorted keys in fixed-size pages. Internal pages contain separator keys and child
pointers. Leaf pages contain keys and record references or inline values. A lookup descends from
the root to one leaf, so its page count grows with the tree height rather than the number of
records.

Inserting into a full page causes a split. The engine allocates a sibling, divides the keys, and
inserts a separator into the parent. The root can also split, which adds one level to the tree.
Concurrent implementations protect structural changes with latches or optimistic validation.

## MVCC snapshots

Multi-version concurrency control stores enough history for readers to see a stable snapshot.
A row version records the transaction that created it and, when applicable, the transaction that
replaced or deleted it. Visibility rules compare those identifiers with the reader's snapshot.

Old versions cannot remain forever. A vacuum or compaction process removes a version after no
active snapshot can see it. A long-running transaction delays that cleanup and can increase
storage use even when the transaction performs no writes.

## LSM compaction

A log-structured merge tree collects recent writes in memory and flushes sorted runs to disk.
Reads may need to check several runs, so bloom filters and sparse indexes avoid many disk reads.
Compaction merges runs, removes overwritten values, and restores bounds on read amplification.

Compaction trades background write work for cheaper future reads. Size-tiered compaction writes
less data during ingestion but can leave more runs to search. Leveled compaction limits overlap
between runs but rewrites data more often.

