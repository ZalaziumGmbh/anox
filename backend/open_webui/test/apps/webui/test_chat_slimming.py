"""Tests for chat response slimming and history repair.

These functions protect against browser freezes from large chat JSON payloads
by stripping unused data from API responses without removing any messages.
"""

import copy
import json
import pytest

from open_webui.models.chats import repair_chat_history, slim_chat_sources


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(doc_count=25, doc_len=5000, include_user=True):
    """Build a realistic RAG source entry."""
    entry = {
        "source": {
            "id": "src-1",
            "name": "Test Knowledge",
            "url": "/api/v1/knowledge/src-1",
            "embed_url": "/api/v1/knowledge/src-1/embed",
            "type": "collection",
        },
        "document": [f"chunk-{i} " + ("x" * doc_len) for i in range(doc_count)],
        "metadata": [{"hash": f"hash-{i}", "name": f"doc-{i}.pdf"} for i in range(doc_count)],
        "distances": [0.8 - i * 0.01 for i in range(doc_count)],
    }
    if include_user:
        entry["source"]["user_id"] = "user-1"
        entry["source"]["user"] = {"id": "user-1", "name": "Test", "extra": "x" * 40000}
        entry["source"]["description"] = "A test knowledge base"
        entry["source"]["meta"] = None
        entry["source"]["access_control"] = {"read": {"group_ids": []}}
        entry["source"]["created_at"] = 1700000000
        entry["source"]["updated_at"] = 1700000000
        entry["source"]["files"] = [{"id": "f1", "filename": "doc.pdf"}]
    return entry


def _make_chat(msg_count=10, with_sources=True, with_base64=False):
    """Build a realistic chat dict with history and top-level messages."""
    messages_dict = {}
    messages_list = []
    prev_id = None

    for i in range(msg_count):
        mid = f"msg-{i}"
        role = "user" if i % 2 == 0 else "assistant"
        msg = {
            "id": mid,
            "parentId": prev_id,
            "childrenIds": [],
            "role": role,
            "content": f"Message {i} content",
        }
        if role == "assistant" and with_sources:
            msg["sources"] = [_make_source(), _make_source()]
        if i == 0 and with_base64:
            msg["files"] = [
                {"type": "image", "url": "data:image/png;base64," + "A" * 100000}
            ]
        if prev_id and prev_id in messages_dict:
            messages_dict[prev_id]["childrenIds"].append(mid)
        messages_dict[mid] = msg
        messages_list.append(copy.deepcopy(msg))
        prev_id = mid

    return {
        "history": {
            "currentId": prev_id,
            "messages": messages_dict,
        },
        "messages": messages_list,
        "title": "Test Chat",
        "models": ["gpt-4"],
    }


# ---------------------------------------------------------------------------
# slim_chat_sources tests
# ---------------------------------------------------------------------------

class TestSlimChatSources:
    """Tests for the slim_chat_sources function."""

    def test_preserves_all_messages(self):
        """No messages should be removed — only payload reduced."""
        chat = _make_chat(msg_count=20, with_sources=True)
        result = slim_chat_sources(chat)

        orig_ids = set(chat["history"]["messages"].keys())
        result_ids = set(result["history"]["messages"].keys())
        assert orig_ids == result_ids, "history.messages IDs must match"

        assert len(result["messages"]) == len(chat["messages"]), \
            "top-level messages count must match"

    def test_preserves_message_content(self):
        """Message text content must not be altered."""
        chat = _make_chat(msg_count=6)
        result = slim_chat_sources(chat)

        for mid in chat["history"]["messages"]:
            orig = chat["history"]["messages"][mid]["content"]
            slimmed = result["history"]["messages"][mid]["content"]
            assert orig == slimmed, f"Content changed for {mid}"

    def test_preserves_parent_chain(self):
        """The parentId chain must remain fully intact."""
        chat = _make_chat(msg_count=20)
        result = slim_chat_sources(chat)

        current_id = result["history"]["currentId"]
        assert current_id == chat["history"]["currentId"]

        chain = 0
        mid = current_id
        visited = set()
        while mid:
            assert mid not in visited, "Circular reference"
            visited.add(mid)
            assert mid in result["history"]["messages"], f"{mid} missing"
            chain += 1
            mid = result["history"]["messages"][mid].get("parentId")

        assert chain == 20, f"Chain should be 20, got {chain}"

    def test_strips_source_user_object(self):
        """The source.user field (42KB+) must be removed."""
        chat = _make_chat(msg_count=4, with_sources=True)
        result = slim_chat_sources(chat)

        for mid, msg in result["history"]["messages"].items():
            for src in msg.get("sources", []):
                assert "user" not in src["source"], \
                    "source.user should be stripped"
                assert "user_id" not in src["source"]
                assert "description" not in src["source"]
                assert "access_control" not in src["source"]

    def test_keeps_essential_source_fields(self):
        """id, name, url, embed_url, type must be preserved."""
        chat = _make_chat(msg_count=4, with_sources=True)
        result = slim_chat_sources(chat)

        for mid, msg in result["history"]["messages"].items():
            for src in msg.get("sources", []):
                s = src["source"]
                assert s.get("id") == "src-1"
                assert s.get("name") == "Test Knowledge"
                assert s.get("type") == "collection"

    def test_limits_document_chunks(self):
        """Document arrays should be limited to 5 chunks."""
        chat = _make_chat(msg_count=4, with_sources=True)
        result = slim_chat_sources(chat)

        for mid, msg in result["history"]["messages"].items():
            for src in msg.get("sources", []):
                docs = src.get("document", [])
                assert len(docs) <= 5, f"Got {len(docs)} doc chunks"
                meta = src.get("metadata", [])
                assert len(meta) <= len(docs), "metadata should match docs"
                dists = src.get("distances", [])
                assert len(dists) <= len(docs), "distances should match docs"

    def test_truncates_long_doc_chunks(self):
        """Document chunks longer than 2000 chars should be truncated."""
        chat = _make_chat(msg_count=4, with_sources=True)
        result = slim_chat_sources(chat)

        for mid, msg in result["history"]["messages"].items():
            for src in msg.get("sources", []):
                for doc in src.get("document", []):
                    assert len(doc) <= 2000, f"Doc chunk too long: {len(doc)}"

    def test_strips_base64_from_top_messages_only(self):
        """Base64 data URLs stripped from top-level messages but kept in history."""
        chat = _make_chat(msg_count=4, with_sources=False, with_base64=True)
        result = slim_chat_sources(chat)

        # history.messages should keep base64
        hist_files = result["history"]["messages"]["msg-0"].get("files", [])
        assert len(hist_files) == 1
        assert hist_files[0]["url"].startswith("data:"), \
            "history.messages should keep base64"

        # top-level messages should strip base64
        top_files = result["messages"][0].get("files", [])
        assert len(top_files) == 1
        assert top_files[0]["url"] == "", \
            "top messages should strip base64"

    def test_reduces_payload_size(self):
        """Slimmed payload should be significantly smaller."""
        chat = _make_chat(msg_count=20, with_sources=True)
        orig_size = len(json.dumps(chat))
        result = slim_chat_sources(chat)
        slim_size = len(json.dumps(result))

        assert slim_size < orig_size * 0.3, \
            f"Expected >70% reduction, got {(1-slim_size/orig_size)*100:.0f}%"

    def test_no_mutation_of_original(self):
        """Original chat dict must not be modified."""
        chat = _make_chat(msg_count=6, with_sources=True)
        original_json = json.dumps(chat, sort_keys=True)
        slim_chat_sources(chat)
        after_json = json.dumps(chat, sort_keys=True)
        assert original_json == after_json, "Original dict was mutated"

    def test_passthrough_small_chat(self):
        """Chat without sources should pass through unchanged."""
        chat = _make_chat(msg_count=4, with_sources=False, with_base64=False)
        result = slim_chat_sources(chat)
        assert result is chat, "Should return same object when nothing to slim"

    def test_handles_missing_history(self):
        """Should not crash on chat dict without history."""
        chat = {"title": "Empty", "messages": []}
        result = slim_chat_sources(chat)
        assert result is chat

    def test_handles_empty_sources(self):
        """Messages with empty sources list should pass through."""
        chat = _make_chat(msg_count=4, with_sources=False)
        chat["history"]["messages"]["msg-1"]["sources"] = []
        result = slim_chat_sources(chat)
        assert result is chat

    def test_sources_not_a_list(self):
        """If sources is a non-list value, should not crash."""
        chat = _make_chat(msg_count=4, with_sources=False)
        chat["history"]["messages"]["msg-1"]["sources"] = "invalid"
        result = slim_chat_sources(chat)
        assert result is chat

    def test_source_entry_not_a_dict(self):
        """If a source entry is not a dict (e.g. None), should skip it."""
        chat = _make_chat(msg_count=4, with_sources=False)
        chat["history"]["messages"]["msg-1"]["sources"] = [None, "string", 42]
        result = slim_chat_sources(chat)
        assert result is chat

    def test_source_without_source_obj(self):
        """Source entry without 'source' key should not crash."""
        chat = _make_chat(msg_count=4, with_sources=False)
        chat["history"]["messages"]["msg-1"]["sources"] = [
            {"document": ["text"], "metadata": [{"name": "a"}], "distances": [0.5]}
        ]
        result = slim_chat_sources(chat)
        # Should still process document truncation
        assert "msg-1" in result["history"]["messages"]

    def test_document_not_a_list(self):
        """If document field is not a list, should not crash."""
        chat = _make_chat(msg_count=4, with_sources=False)
        chat["history"]["messages"]["msg-1"]["sources"] = [
            {"source": {"id": "1", "name": "x"}, "document": "not-a-list"}
        ]
        result = slim_chat_sources(chat)
        assert "msg-1" in result["history"]["messages"]

    def test_no_top_level_messages(self):
        """Chat without top-level messages list should work fine."""
        chat = _make_chat(msg_count=4, with_sources=True)
        del chat["messages"]
        result = slim_chat_sources(chat)
        assert "messages" not in result
        # Sources in history should still be slimmed
        for mid, msg in result["history"]["messages"].items():
            for src in msg.get("sources", []):
                assert "user" not in src.get("source", {})

    def test_files_with_short_data_url_not_stripped(self):
        """Small data URLs (< 1000 chars like icons) should be kept."""
        chat = _make_chat(msg_count=2, with_sources=False)
        small_data_url = "data:image/png;base64," + "A" * 100  # small
        chat["messages"][0]["files"] = [{"type": "image", "url": small_data_url}]
        result = slim_chat_sources(chat)
        assert result["messages"][0]["files"][0]["url"] == small_data_url

    def test_files_with_server_url_not_stripped(self):
        """Server-side file URLs should never be stripped."""
        chat = _make_chat(msg_count=2, with_sources=False)
        chat["messages"][0]["files"] = [
            {"type": "image", "url": "/api/v1/files/abc123/content"}
        ]
        # Need sources to trigger slimming, add one
        chat["history"]["messages"]["msg-1"]["sources"] = [_make_source()]
        result = slim_chat_sources(chat)
        assert result["messages"][0]["files"][0]["url"] == "/api/v1/files/abc123/content"

    def test_single_message_chat(self):
        """Chat with only 1 message should work."""
        chat = _make_chat(msg_count=1, with_sources=True)
        result = slim_chat_sources(chat)
        assert len(result["history"]["messages"]) == 1

    def test_mixed_messages_with_and_without_sources(self):
        """Only assistant messages have sources; user messages should pass through."""
        chat = _make_chat(msg_count=6, with_sources=True)
        result = slim_chat_sources(chat)
        for mid, msg in result["history"]["messages"].items():
            if msg["role"] == "user":
                assert "sources" not in msg or msg.get("sources") is None
            elif msg["role"] == "assistant":
                for src in msg.get("sources", []):
                    assert "user" not in src.get("source", {})

    def test_concurrent_slim_calls(self):
        """Multiple slim calls on same chat should produce identical results."""
        chat = _make_chat(msg_count=10, with_sources=True)
        r1 = slim_chat_sources(chat)
        r2 = slim_chat_sources(chat)
        assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)

    def test_slim_is_idempotent(self):
        """Slimming an already-slimmed chat should return it unchanged."""
        chat = _make_chat(msg_count=10, with_sources=True)
        first = slim_chat_sources(chat)
        second = slim_chat_sources(first)
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


# ---------------------------------------------------------------------------
# repair_chat_history tests
# ---------------------------------------------------------------------------

class TestRepairChatHistory:
    """Tests for the repair_chat_history function."""

    def test_no_repair_needed(self):
        """Healthy chat should not be modified."""
        chat = _make_chat(msg_count=10, with_sources=False)
        result, repaired = repair_chat_history(chat)
        assert repaired is False
        assert result is chat

    def test_repairs_broken_current_id(self):
        """When currentId parent points to missing message, should repair."""
        chat = _make_chat(msg_count=5, with_sources=False)
        # Add orphan message with broken parent and set it as currentId
        chat["history"]["messages"]["orphan"] = {
            "id": "orphan",
            "parentId": "deleted-msg",
            "childrenIds": [],
            "role": None,
        }
        chat["history"]["currentId"] = "orphan"

        result, repaired = repair_chat_history(chat)
        assert repaired is True
        assert result["history"]["currentId"] == "msg-4", \
            f"Should point to msg-4, got {result['history']['currentId']}"

    def test_repair_preserves_all_messages(self):
        """Repair should not remove any messages."""
        chat = _make_chat(msg_count=5, with_sources=False)
        chat["history"]["messages"]["orphan"] = {
            "id": "orphan",
            "parentId": "deleted-msg",
            "childrenIds": [],
            "role": None,
        }
        chat["history"]["currentId"] = "orphan"

        result, _ = repair_chat_history(chat)
        assert set(result["history"]["messages"].keys()) == \
            set(chat["history"]["messages"].keys())

    def test_no_repair_on_empty_chat(self):
        """Empty chat should not crash."""
        chat = {"history": {"currentId": None, "messages": {}}}
        result, repaired = repair_chat_history(chat)
        assert repaired is False

    def test_no_mutation_of_original(self):
        """Original chat dict must not be modified by repair."""
        chat = _make_chat(msg_count=5, with_sources=False)
        chat["history"]["messages"]["orphan"] = {
            "id": "orphan",
            "parentId": "deleted-msg",
            "childrenIds": [],
            "role": None,
        }
        chat["history"]["currentId"] = "orphan"

        original_cid = chat["history"]["currentId"]
        repair_chat_history(chat)
        assert chat["history"]["currentId"] == original_cid, \
            "Original should not be mutated"

    def test_picks_longest_chain(self):
        """When multiple branches exist, should pick the longest chain."""
        chat = {
            "history": {
                "currentId": "orphan",
                "messages": {
                    # Branch A: 3 messages
                    "a1": {"id": "a1", "parentId": None, "childrenIds": ["a2"]},
                    "a2": {"id": "a2", "parentId": "a1", "childrenIds": ["a3"]},
                    "a3": {"id": "a3", "parentId": "a2", "childrenIds": []},
                    # Branch B: 5 messages
                    "b1": {"id": "b1", "parentId": None, "childrenIds": ["b2"]},
                    "b2": {"id": "b2", "parentId": "b1", "childrenIds": ["b3"]},
                    "b3": {"id": "b3", "parentId": "b2", "childrenIds": ["b4"]},
                    "b4": {"id": "b4", "parentId": "b3", "childrenIds": ["b5"]},
                    "b5": {"id": "b5", "parentId": "b4", "childrenIds": []},
                    # Orphan
                    "orphan": {"id": "orphan", "parentId": "gone", "childrenIds": []},
                },
            }
        }
        result, repaired = repair_chat_history(chat)
        assert repaired is True
        assert result["history"]["currentId"] == "b5", \
            "Should pick the longest chain (b5)"

    def test_current_id_not_in_messages(self):
        """currentId pointing to a non-existent message should trigger repair."""
        chat = _make_chat(msg_count=5, with_sources=False)
        chat["history"]["currentId"] = "completely-gone"
        result, repaired = repair_chat_history(chat)
        assert repaired is True
        assert result["history"]["currentId"] == "msg-4"

    def test_circular_reference(self):
        """Circular parentId chain should not cause infinite loop."""
        chat = {
            "history": {
                "currentId": "a",
                "messages": {
                    "a": {"id": "a", "parentId": "b", "childrenIds": []},
                    "b": {"id": "b", "parentId": "a", "childrenIds": ["a"]},
                },
            }
        }
        # Should not hang — just return without crashing
        result, repaired = repair_chat_history(chat)
        # Both messages form a cycle, chain_len=2 which is >= len(messages)-5
        # So no repair needed
        assert repaired is False

    def test_no_children_ids_field(self):
        """Messages without childrenIds field should not crash repair."""
        chat = {
            "history": {
                "currentId": "orphan",
                "messages": {
                    "m1": {"id": "m1", "parentId": None},
                    "m2": {"id": "m2", "parentId": "m1"},
                    "m3": {"id": "m3", "parentId": "m2"},
                    "orphan": {"id": "orphan", "parentId": "gone"},
                },
            }
        }
        result, repaired = repair_chat_history(chat)
        assert repaired is True
        assert result["history"]["currentId"] == "m3"

    def test_single_message_broken(self):
        """Chat with 1 real message + 1 orphan currentId."""
        chat = {
            "history": {
                "currentId": "orphan",
                "messages": {
                    "m1": {"id": "m1", "parentId": None, "childrenIds": []},
                    "orphan": {"id": "orphan", "parentId": "gone", "childrenIds": []},
                },
            }
        }
        result, repaired = repair_chat_history(chat)
        assert repaired is True
        assert result["history"]["currentId"] == "m1"

    def test_repair_then_slim_integration(self):
        """Repair followed by slim should produce a valid, smaller result."""
        chat = _make_chat(msg_count=10, with_sources=True)
        chat["history"]["messages"]["orphan"] = {
            "id": "orphan",
            "parentId": "deleted",
            "childrenIds": [],
            "role": None,
        }
        chat["history"]["currentId"] = "orphan"

        repaired, was_repaired = repair_chat_history(chat)
        assert was_repaired is True

        slimmed = slim_chat_sources(repaired)
        # All original messages should still be there
        assert len(slimmed["history"]["messages"]) == 11  # 10 + orphan
        # Chain should work from repaired currentId
        chain = 0
        mid = slimmed["history"]["currentId"]
        while mid and mid in slimmed["history"]["messages"]:
            chain += 1
            mid = slimmed["history"]["messages"][mid].get("parentId")
        assert chain == 10
