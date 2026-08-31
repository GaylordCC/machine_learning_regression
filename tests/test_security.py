from machine_learning.core.security import hash_password, verify_password


def test_hash_password_does_not_return_the_plain_value():
    plain = "correcthorsebattery"
    hashed = hash_password(plain)
    assert hashed != plain


def test_verify_password_accepts_the_correct_password():
    plain = "correcthorsebattery"
    hashed = hash_password(plain)
    assert verify_password(plain, hashed) is True


def test_verify_password_rejects_a_wrong_password():
    hashed = hash_password("correcthorsebattery")
    assert verify_password("wrong-password", hashed) is False


def test_hashing_the_same_password_twice_yields_different_hashes():
    """bcrypt salts each hash -- guards against ever "optimizing" this into a
    deterministic hash, which would make identical passwords recognizable."""
    plain = "correcthorsebattery"
    assert hash_password(plain) != hash_password(plain)
