"""Password hashing for the (currently unused) User model.

No endpoint creates or authenticates users yet -- this exists so that when
one is added, it can't accidentally store a plaintext password. See
security-vulnerabilities.md checklist item 3 (secrets) in the audit that
flagged this.

Uses bcrypt directly rather than passlib: passlib 1.7.4 (last released
~2020, effectively unmaintained) is incompatible with current bcrypt
releases (verified here -- raises "password cannot be longer than 72
bytes" even for short passwords, due to passlib's bcrypt-version-sniffing
code breaking against bcrypt>=4.1). bcrypt itself is actively maintained
and is all passlib was wrapping here anyway.
"""
import bcrypt

_ENCODING = "utf-8"


def hash_password(plain_password: str) -> str:
    hashed = bcrypt.hashpw(plain_password.encode(_ENCODING), bcrypt.gensalt())
    return hashed.decode(_ENCODING)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode(_ENCODING), hashed_password.encode(_ENCODING))
