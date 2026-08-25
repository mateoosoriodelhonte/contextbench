# Network protocol notes

## Flow and congestion control

TCP flow control protects the receiver. The receiver advertises how much unread data fits in its
buffer, and the sender keeps unacknowledged bytes within that window. A zero window stops new
payload until the receiver announces space.

Congestion control protects the network path. The sender maintains a congestion window based on
acknowledgments, loss, and timing. The usable send window is the smaller of the receiver window
and the congestion window. These two controls solve different problems even though both limit
in-flight data.

## DNS caching

A recursive resolver asks authoritative servers for records and caches the answers. Each record
has a time to live. The resolver can reuse the answer until that time expires, which reduces
latency and load but delays how quickly a record change reaches every client.

Negative caching stores proof that a name or record type does not exist. It prevents repeated
queries for the same missing data. The negative cache duration comes from the zone's authority
data rather than from a made-up client default.

## HTTP retries

A client can safely retry an idempotent operation when the same request has the same intended
effect after one or many executions. GET, PUT, and DELETE have idempotent semantics, although a
server can still implement them badly. POST is not idempotent by default.

An idempotency key lets a server recognize repeated POST attempts. The server stores the key and
the first result inside a defined retention period. Later requests with the same key and payload
return that result instead of creating another resource. A reused key with a different payload
must fail rather than silently selecting one request.

Retries need backoff and a limit. Immediate retries can add load to a service that is already
failing. Exponential backoff spreads attempts over time, and random jitter keeps many clients
from retrying on the same schedule.
