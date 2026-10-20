"""Outbound attachment delivery: @@attach upload + marker stripping in the daemon."""

import asyncio
import types
from unittest.mock import AsyncMock, patch

from slack_daemon import SlackDaemon, _meta_context_block, _reply_footer_blocks


def _run_deliver(channel, thread_ts, response, *, fake_post, upload_return):
    """Build a SlackDaemon INSIDE the loop, stub its post + upload_files, and
    run _deliver_response. Returns the patched upload_files mock for assertions.

    SlackDaemon.__init__ builds an AsyncSocketModeHandler that creates an
    aiohttp.ClientSession, which requires a running event loop — so the daemon
    MUST be constructed inside asyncio.run, not at module/helper top level.
    self._app.client is a read-only property, so outbound uploads are exercised
    by patching slack_daemon.attachments.upload_files rather than the client.
    """
    async def _main():
        with patch(
            "slack_daemon.attachments.upload_files",
            new=AsyncMock(return_value=upload_return),
        ) as up:
            d = SlackDaemon(bot_token="xoxb-test", app_token="xapp-test")
            d._post_response = fake_post  # type: ignore[assignment]
            await d._deliver_response(channel, thread_ts, response)
            return up

    return asyncio.run(_main())


class TestDeliverResponse:
    def test_uploads_existing_path_strips_marker_and_posts_remainder(self, tmp_path):
        posted = {}

        async def fake_post(channel, thread_ts, text, footer_blocks=None):
            posted["channel"] = channel
            posted["thread_ts"] = thread_ts
            posted["text"] = text

        chart = tmp_path / "chart.png"
        chart.write_bytes(b"PNG")
        reply = f"Here is the chart.\n@@attach {chart}\nDone."

        up = _run_deliver(
            "C1", "T-1", reply, fake_post=fake_post, upload_return=[str(chart)]
        )

        up.assert_awaited_once()
        kwargs = up.await_args.kwargs
        # _deliver_response passes the client positionally and paths as a kwarg.
        assert kwargs["paths"] == [str(chart)]
        assert kwargs["channel"] == "C1"
        assert kwargs["thread_ts"] == "T-1"
        assert posted["text"] == "Here is the chart.\nDone."
        assert "@@attach" not in posted["text"]

    def test_nonexistent_path_still_attempted_but_text_stripped(self, tmp_path):
        # _deliver_response always delegates to upload_files when markers exist;
        # the *existence* skip lives in upload_files itself (covered in Task 3).
        # Here we assert the marker line is stripped from the posted text even
        # when the path is bogus, and upload_files is still called with it.
        posted = {}

        async def fake_post(channel, thread_ts, text, footer_blocks=None):
            posted["text"] = text

        missing = tmp_path / "nope.png"  # never created
        reply = f"text before\n@@attach {missing}\ntext after"

        up = _run_deliver(
            "C1", "T-2", reply, fake_post=fake_post, upload_return=[]
        )

        up.assert_awaited_once()
        assert up.await_args.kwargs["paths"] == [str(missing)]
        assert posted["text"] == "text before\ntext after"

    def test_no_markers_posts_full_text_no_upload(self):
        posted = {}

        async def fake_post(channel, thread_ts, text, footer_blocks=None):
            posted["text"] = text

        up = _run_deliver(
            "C1", "T-3", "just a normal reply", fake_post=fake_post, upload_return=[]
        )

        up.assert_not_awaited()
        assert posted["text"] == "just a normal reply"

    def test_reply_only_marker_posts_nothing(self, tmp_path):
        post_calls = []

        async def fake_post(channel, thread_ts, text, footer_blocks=None):
            post_calls.append(text)

        chart = tmp_path / "only.png"
        chart.write_bytes(b"X")

        up = _run_deliver(
            "C1", "T-4", f"@@attach {chart}", fake_post=fake_post,
            upload_return=[str(chart)],
        )

        up.assert_awaited_once()
        assert post_calls == []  # no text remained, so no chat_postMessage


def _post_with_capture(footer_blocks):
    """Run the REAL _post_response with a capturing client; return posted kwargs."""
    async def _main():
        posts: list[dict] = []

        class _C:
            async def chat_postMessage(self, **kw):
                posts.append(kw)
                return {"ts": "1"}

        d = SlackDaemon(bot_token="xoxb-test", app_token="xapp-test")
        d._app = types.SimpleNamespace(client=_C())  # stub the bolt app's client
        await d._post_response("C1", "T1", "hello world", footer_blocks=footer_blocks)
        return posts

    return asyncio.run(_main())


class _FakeProg:
    def __init__(self, meta="", files=""):
        self.meta = meta
        self.files = files


class _FakeReporter:
    def __init__(self, progress):
        self.last_progress = progress


class TestReplyFooter:
    def test_footer_blocks_appended_under_last_message(self):
        footer = [
            {"type": "section", "text": {"type": "mrkdwn", "text": "📝 *2 files changed*"}},
            _meta_context_block("3 tools · 1m02s"),
        ]
        blocks = _post_with_capture(footer)[-1]["blocks"]
        assert blocks[-2]["type"] == "section" and "2 files changed" in blocks[-2]["text"]["text"]
        assert blocks[-1]["type"] == "context"
        assert blocks[-1]["elements"][0]["text"] == "3 tools · 1m02s"

    def test_no_footer_means_only_markdown_blocks(self):
        blocks = _post_with_capture([])[-1]["blocks"]
        assert all(b["type"] == "markdown" for b in blocks)

    def test_meta_context_block_helper(self):
        assert _meta_context_block("") is None
        blk = _meta_context_block("a · b")
        assert blk["type"] == "context" and blk["elements"][0]["text"] == "a · b"

    def test_footer_from_reporter_has_files_then_meta_with_subagents_and_skills(self):
        r = _FakeReporter(_FakeProg(
            meta="3 tools · 🤖 1 subagent · 🧩 brainstorming",
            files="📝 *1 file changed*\n• `x.py`",
        ))
        blocks = _reply_footer_blocks(r)
        assert blocks[0]["type"] == "section" and "1 file changed" in blocks[0]["text"]["text"]
        meta_text = blocks[1]["elements"][0]["text"]
        assert "subagent" in meta_text and "brainstorming" in meta_text  # counts + skills

    def test_footer_empty_without_progress(self):
        assert _reply_footer_blocks(None) == []
        assert _reply_footer_blocks(_FakeReporter(None)) == []
