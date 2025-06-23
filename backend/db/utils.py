# utils.py
import hashlib

def name_to_subject_id(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:8]

def subject_id_to_name(subject_id: str) -> str:
    # Store original name in DB for reverse lookup
    raise NotImplementedError("subject_id is one-way; store mapping instead")
