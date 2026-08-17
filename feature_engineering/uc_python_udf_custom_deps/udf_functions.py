"""UDF function definitions for Unity Catalog registration.

Define all UC Python UDF functions here. The notebook imports this module,
extracts function bodies via `inspect`, and registers them in Unity Catalog.

Each function's name must match the desired UC function name (without catalog/schema prefix).
"""


# ─── UDF: udf_dump_json ────────────────────────────────────────────────────────
def udf_dump_json(data):
    import simplejson as json
    try:
        return json.dumps({"result": data, "length": len(data) if data else 0})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ─── UDF: udf_sha3_hash ───────────────────────────────────────────────────────
def udf_sha3_hash(data):
    if data is None:
        return None
    from Crypto.Hash import SHA3_256
    h = SHA3_256.new()
    h.update(data.encode("utf-8"))
    return h.hexdigest()


# ─── UDF: udf_mask_email ──────────────────────────────────────────────────────
def udf_mask_email(email):
    if email is None:
        return None
    parts = email.split("@", 1)
    if len(parts) != 2:
        return email
    username, domain = parts
    if len(username) <= 2:
        masked = "*" * len(username)
    else:
        masked = username[0] + "*" * (len(username) - 2) + username[-1]
    return f"{masked}@{domain}"


# ─── UDF Metadata Registry ────────────────────────────────────────────────────
# Maps each function to its SQL parameter signature and return type.
# The notebook uses this to build CREATE OR REPLACE FUNCTION statements.

UDF_DEFINITIONS = [
    {
        "func": udf_dump_json,
        "params": "data STRING",
        "returns": "STRING",
        "comment": "Serializes a string value to a JSON object using simplejson.",
    },
    {
        "func": udf_sha3_hash,
        "params": "data STRING",
        "returns": "STRING",
        "comment": "Returns the SHA3-256 hex digest of the input string using pycryptodome.",
    },
    {
        "func": udf_mask_email,
        "params": "email STRING",
        "returns": "STRING",
        "comment": "Masks the local part of an email address, preserving first/last char and domain.",
    },
]
