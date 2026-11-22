"""In-memory conversation store: one GiftConcierge (dialogue state) per session id.

Turns within a session run one at a time (per-session lock). Idle sessions are
dropped after a TTL, and the least recently used session is evicted when the
store is full. State lives in this process only: restarting the server or
running several workers starts conversations afresh.
"""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class Session:
    concierge: object
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_used: float = field(default_factory=time.monotonic)
    has_results: bool = False


class SessionStore:
    def __init__(self, factory, ttl_seconds=7200, max_sessions=200):
        self._factory = factory
        self._ttl = ttl_seconds
        self._max = max_sessions
        self._sessions = OrderedDict()
        self._lock = threading.Lock()

    def get(self, session_id):
        with self._lock:
            self._expire()
            session = self._sessions.get(session_id)
            if session is None:
                session = Session(self._factory())
                self._sessions[session_id] = session
                while len(self._sessions) > self._max:
                    self._sessions.popitem(last=False)
            self._sessions.move_to_end(session_id)
            session.last_used = time.monotonic()
            return session

    def drop(self, session_id):
        with self._lock:
            self._sessions.pop(session_id, None)

    def __len__(self):
        return len(self._sessions)

    def _expire(self):
        cutoff = time.monotonic() - self._ttl
        for sid in [sid for sid, s in self._sessions.items() if s.last_used < cutoff]:
            del self._sessions[sid]
