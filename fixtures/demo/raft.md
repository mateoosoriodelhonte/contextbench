# Raft replication notes

## Leader election

A Raft server starts as a follower. The follower expects periodic AppendEntries calls from the
current leader. If the election timer expires before a valid leader message arrives, the server
increments its term, becomes a candidate, votes for itself, and requests votes from the other
servers. A candidate becomes leader after it receives votes from a majority of the cluster.

Election timers use randomized deadlines. If every follower used the same timeout, several
servers could become candidates at once and split the vote. Random deadlines make one candidate
more likely to ask for votes before the others. A new term starts another election if no
candidate wins.

## Log replication

The leader accepts a client command, appends it to its local log, and sends AppendEntries calls
to followers. Each call names the term and index immediately before the new entries. A follower
accepts the entries only when its own log has the same term at that position. This consistency
check prevents two different commands from occupying the same committed log position.

An entry is committed after the leader knows that a majority has stored it and the entry belongs
to the leader's current term. The leader then applies committed entries to its state machine in
index order. Followers learn the commit index from later AppendEntries calls and apply the same
prefix.

## A follower that falls behind

The leader tracks `nextIndex` and `matchIndex` for every follower. `nextIndex` is the next log
position the leader plans to send. `matchIndex` is the highest position known to exist on that
follower.

If a follower rejects AppendEntries because the preceding index or term does not match, the
leader moves that follower's `nextIndex` backward and retries. Once it finds a matching prefix,
the leader sends the missing suffix. The follower removes conflicting uncommitted entries and
copies the leader's entries. Other followers keep making progress; one slow replica does not
block a majority from committing.

A follower can be too far behind for the leader to retain the needed log prefix. The leader then
sends an InstallSnapshot call. The follower installs the snapshot, discards log entries covered
by it, and resumes normal AppendEntries replication after the snapshot's last included index.

## Read safety

A leader cannot answer every read from memory without checking its authority. It may have been
separated from a newer leader. A read-index protocol confirms contact with a majority in the
current term before the state machine serves a linearizable read. A lease can reduce the number
of messages, but it depends on bounded clock behavior and requires a carefully stated timing
assumption.
