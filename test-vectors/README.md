# Field 15 — calibration-provenance test vectors

`field15_signing.py` regenerates `field15_vectors.json`: the calibration-profile
digest (SHA-256 over RFC 8949 deterministic CBOR) and the domain-separated PoH
signing vectors from draft-ayerbe-trip-protocol (Calibration Provenance).

```
python3 -I field15_signing.py          # human-readable
python3 -I field15_signing.py --json   # regenerate field15_vectors.json
```

Uses an all-zero Ed25519 test seed (not a real key). The script asserts that the
signature verifies, that tampering with field 7 or dropping the domain-separation
label both break verification, and that an inline-profile digest mismatch is
rejected.
