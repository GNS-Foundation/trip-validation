#!/usr/bin/env python3
"""Field 15 (calibration provenance) reproducible test vectors.

Generates the calibration-profile digest and the PoH signing vectors used in
draft-ayerbe-trip-protocol (Calibration Provenance). Uses an ALL-ZERO Ed25519
test seed -- NOT a real key. Run: python3 -I test-vectors/field15_signing.py
"""
import cbor2, hashlib, json, sys
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.exceptions import InvalidSignature

DOMAIN = b"TRIP-PoH-Certificate-v1"

profile = {
  0:1, 1:"log-periodogram-gph-1983", 2:10, 3:"same-cell-consecutive",
  4:900, 5:256, 6:[0.00390625,0.5], 7:0.70, 8:0.55,
  9:"separability-weighted", 10:"geolife-1.3", 11:1730160000,
}
prof_cbor = cbor2.dumps(profile, canonical=True)
digest = hashlib.sha256(prof_cbor).digest()

sk = Ed25519PrivateKey.from_private_bytes(bytes(32))  # TEST seed
pk = sk.public_key()
cert = {
  0:bytes.fromhex("11"*32),1:1730160000,2:42,3:0.68,4:1.75,5:3.2,6:0.88,
  7:0.81,8:0.73,9:57,10:240,11:86400,
  12:bytes.fromhex("22"*16),13:bytes.fromhex("33"*32),
  15:{0:digest,1:profile},
}
signing_input = DOMAIN + cbor2.dumps(cert, canonical=True)
sig = sk.sign(signing_input)

checks = {}
try: pk.verify(sig, signing_input); checks["sig_verifies"]=True
except InvalidSignature: checks["sig_verifies"]=False
ct=dict(cert); ct[7]=0.99
try: pk.verify(sig, DOMAIN+cbor2.dumps(ct,canonical=True)); checks["tamper_field7_rejected"]=False
except InvalidSignature: checks["tamper_field7_rejected"]=True
try: pk.verify(sig, cbor2.dumps(cert,canonical=True)); checks["domain_sep_enforced"]=False
except InvalidSignature: checks["domain_sep_enforced"]=True
checks["inline_digest_matches"]= hashlib.sha256(cbor2.dumps(cert[15][1],canonical=True)).digest()==cert[15][0]
wp=dict(profile); wp[7]=0.50
checks["wrong_inline_rejected"]= hashlib.sha256(cbor2.dumps(wp,canonical=True)).digest()!=cert[15][0]

out = {
  "domain_separation": DOMAIN.decode(),
  "deterministic_cbor": "RFC 8949 section 4.2.1 (core deterministic)",
  "example_calibration_profile": profile,
  "profile_cbor_hex": prof_cbor.hex(),
  "profile_digest_sha256": digest.hex(),
  "verifier_public_key_ed25519": pk.public_bytes_raw().hex(),
  "signature_field14": sig.hex(),
  "checks": checks,
  "note": "All-zero Ed25519 test seed; not a real key.",
}
if "--json" in sys.argv:
    print(json.dumps(out, indent=2))
else:
    for k,v in out.items():
        if k!="checks": print(f"{k}: {v}")
    print("checks:", checks)
assert all(checks.values()), "a guard did not hold"
