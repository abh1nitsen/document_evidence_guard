"""Bounded search hypotheses, never facts or permission to release an answer.

Only the extractive index uses these tokens. Source text, offsets, original
questions and the verifier's role/scope checks are untouched.
"""

import re

from .evidence import terms

PROTOCOL = "relationship-retrieval-v1"
SPELLINGS = {
    "organiser": "organizer",
    "organisers": "organizers",
    "organise": "organize",
    "organises": "organizes",
    "organised": "organized",
    "organising": "organizing",
}
FILLER = terms("who here please could would you we")

# These expressions discover candidates. E.g. hosting a server is not proof
# of organizing an event; the judge must still check the original relationship.
RELATIONSHIPS = {
    "organizer": (
        r"\borgani[sz](?:er|ers|ed|ing|es|e)\b",
        "organized by",
        "run by",
        "hosted by",
    ),
    "sponsor": (r"\bsponsors?(?:ed|ing)?\b", "sponsored by", "funded by", "supported by"),
    "recipient": (
        r"\brecipient\b|\b(?:made|prepared|written) for\b",
        "prepared for",
        "addressed to",
        "recipient",
    ),
    "author": (
        r"\bauthors?\b|\b(?:written|authored) by\b",
        "written by",
        "authored by",
        "author",
    ),
    "publisher": (r"\bpublishers?\b|\bpublished by\b", "published by", "publisher"),
    "owner": (r"\bowners?\b|\bowned by\b", "owned by", "owner"),
    "operator": (r"\boperators?\b|\boperated by\b", "operated by", "operator"),
    "host": (r"\bhosts?\b|\bhosted by\b", "hosted by", "host"),
}


def search_terms(value):
    normalized = re.sub(r"\b\w+\b", lambda m: SPELLINGS.get(m[0], m[0]), value.casefold())
    return terms(normalized) - FILLER


def expression_matches(expression, text):
    return bool(
        re.search(
            r"\b" + r"\s+".join(map(re.escape, expression.split())) + r"\b", text, re.I
        )
    )


def lexical_query(value, roles):
    """Remove identity-question scaffolding, not subject/scope constraints.

    The raw query remains in the record and all semantic checks. Avoid ranking
    unrelated text on 'made' or repeated 'document' headers in recipient queries.
    """
    if roles:
        value = re.sub(
            r"\bwho (?:is|are) (?:the )?(?:individual|person|entity)\b",
            " ",
            value,
            flags=re.I,
        )
        value = re.sub(
            r"\b(?:this|the selected) (?:document|paper|report)\b",
            " ",
            value,
            flags=re.I,
        )
        if "recipient" in roles:
            value = re.sub(
                r"\b(?:made|prepared|written) for\b", " recipient ", value, flags=re.I
            )
    return " ".join(value.split())


def search_plan(question, query):
    roles = [
        name
        for name, values in RELATIONSHIPS.items()
        if re.search(values[0], question, re.I)
    ]
    alternatives = list(
        dict.fromkeys(
            expression for role in roles for expression in RELATIONSHIPS[role][1:]
        )
    )[:6]
    original_search = lexical_query(question, roles)
    prepared_search = lexical_query(query, roles)
    return {
        "protocol": PROTOCOL,
        "original_question": question,
        "prepared_query": query,
        "relationships": roles,
        "alternatives": alternatives,
        "alternative_search": "exact_phrase_with_original_subject_ranking",
        "queries": list(dict.fromkeys([original_search, prepared_search])),
    }


def match_signals(passage, plan):
    words = search_terms(passage.text)
    roles = plan["relationships"]
    original = search_terms(lexical_query(plan["original_question"], roles))
    flat = " ".join(passage.text.casefold().split())
    relations = [
        role
        for role in roles
        if re.search(RELATIONSHIPS[role][0], flat, re.I)
        or any(expression_matches(s, flat) for s in RELATIONSHIPS[role][1:])
    ]
    return {
        "passage_id": passage.id,
        "content_matches": sorted(words & original),
        "prepared_matches": sorted(
            words & search_terms(lexical_query(plan["prepared_query"], roles))
        ),
        "relationship_matches": relations,
        "context_only": not bool(words & original or relations),
    }
